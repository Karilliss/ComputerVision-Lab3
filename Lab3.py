"""ЛР_3."""
import cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

U = ""
SRC = {"Bing":      (U + "titicaca_bing.png",      (100, 965, 5, 1830)),   
       "MapCarta": (U + "titicaca_mapcarta1.png", (60, 1020, 5, 1770))}

def load(path, crop):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(
            f"Не вдалося завантажити файл: '{path}'. "
        )
    y0, y1, x0, x1 = crop
    return im[y0:y1, x0:x1]

def gray_world(im):
    f = im.astype(np.float32); m = f.reshape(-1, 3).mean(0)
    return np.clip(f * (m.mean() / m), 0, 255).astype(np.uint8)

def clahe_L(im, clip=2.0):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clip, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess(im):
    """фільтрація: білатеральний  + медіанний """
    return cv2.medianBlur(cv2.bilateralFilter(im, 9, 40, 9), 7)

def water_mask(im):
    """Сегментація водних об'єктів у HSV (повне з'єднання через морфологію)"""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower_water = np.array([0, 0, 0])
    upper_water = np.array([180, 255, 75])
    m = cv2.inRange(hsv, lower_water, upper_water)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    
    border = 50
    m[:border, :] = 0
    m[-border:, :] = 0
    m[:, :border] = 0
    m[:, -border:] = 0
    
    blue = im[..., 0].astype(np.float32) - np.maximum(im[..., 1], im[..., 2]).astype(np.float32)
    return m, blue

def contours_of(mask, min_area=1000):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = mask.shape[:2]
    valid_contours = []
    for c in cs:
        area = cv2.contourArea(c)
        if min_area < area < 0.6 * (h_img * w_img):
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if 0.2 * w_img < cX < 0.7 * w_img and 0.1 * h_img < cY < 0.9 * h_img:
                    valid_contours.append(c)
    return sorted(valid_contours, key=cv2.contourArea, reverse=True)

def main_lake(mask):
    cs = contours_of(mask)
    if not cs:
        return [], []

    lake_parts = cs[:2] if len(cs) >= 2 else cs[:1]
    lake_ids = {id(lp) for lp in lake_parts}
    others = [c for c in contours_of(mask, min_area=500) if id(c) not in lake_ids]
    return lake_parts, others

def features(c):
    A = cv2.contourArea(c); P = cv2.arcLength(c, True)
    hull = cv2.convexHull(c); sol = A / cv2.contourArea(hull)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c); ar = max(w, h) / min(w, h)
    circ = 4 * np.pi * A / P ** 2
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return dict(aspect=ar, circularity=circ, solidity=sol, hu=hu, area=A)

def fourier_desc(c, n=32):
    z = c[:, 0, 0] + 1j * c[:, 0, 1]
    d = np.r_[0, np.cumsum(np.abs(np.diff(z)))]; t = np.linspace(0, d[-1], 512, endpoint=False)
    z = np.interp(t, d, z.real) + 1j * np.interp(t, d, z.imag)
    F = np.fft.fft(z - z.mean()); mag = np.abs(F)
    mag = mag / mag[1]
    return np.r_[mag[2:n + 1], mag[-n:-1]]

def compare(c1, c2):
    return dict(matchShapes=cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0),
                fourier_L2=float(np.linalg.norm(fourier_desc(c1) - fourier_desc(c2))))

def iou_aligned(c1, c2, shape):
    def norm(c):
        pts = c[:, 0, :].astype(np.float32); m = cv2.moments(c); cx, cy = m["m10"]/m["m00"], m["m01"]/m["m00"]
        pts -= [cx, cy]; pts /= np.sqrt(cv2.contourArea(c))
        ev, evec = np.linalg.eigh(np.cov(pts.T)); pts = pts @ evec[:, ::-1]
        return pts
    out = []
    for flip in (1, -1):
        a, b = norm(c1), norm(c2) * [flip, 1]
        S = 400; ma = np.zeros((S, S), np.uint8); mb = ma.copy()
        for p, mm in ((a, ma), (b, mb)):
            cv2.fillPoly(mm, [(p * 120 + S / 2).astype(np.int32)], 1)
        for fl2 in (0, 1):
            mb2 = mb[::-1] if fl2 else mb
            out.append((ma & mb2).sum() / (ma | mb2).sum())
    return max(out)

if __name__ == "__main__":
    res, cont, masks, ims = {}, {}, {}, {}
    for name, (p, crop) in SRC.items():
        raw = load(p, crop)
        corr = clahe_L(raw)
        m, blue = water_mask(corr)
        lakes, others = main_lake(m)
        main_c = np.vstack(lakes) if len(lakes) > 1 else lakes[0]
        res[name] = (features(main_c), lakes, others); ims[name] = (raw, corr, m, blue)

    rows = []
    for n, (f, c, o) in res.items():
        rows.append(dict(image=n, area_px=f["area"], aspect=f["aspect"], circularity=f["circularity"],
                         solidity=f["solidity"], **{f"Hu{i+1}": f["hu"][i] for i in range(4)}))
    df = pd.DataFrame(rows).round(4); print(df.to_string(index=False)); df.to_csv("features.csv", index=False)

    cB_parts, cM_parts = res["Bing"][1], res["MapCarta"][1]
    cB = np.vstack(cB_parts) if len(cB_parts) > 1 else cB_parts[0]
    cM = np.vstack(cM_parts) if len(cM_parts) > 1 else cM_parts[0]
    
    same = compare(cB, cM); same["IoU_aligned"] = iou_aligned(cB, cM, None)
    print("\nТітікака(Bing) vs Тітікака(MapCarta):", {k: round(v, 4) for k, v in same.items()})
    
    neg = []
    for n, (f, c, o) in res.items():
        for i, oc in enumerate(o[:6]):
            ref = cM if n == "Bing" else cB
            d = compare(ref, oc); d["IoU_aligned"] = iou_aligned(ref, oc, None)
            d.update(src=n, rank=i + 2, area=cv2.contourArea(oc)); neg.append(d)
    nd = pd.DataFrame(neg).round(4); print("\nНегативні приклади:\n", nd.to_string(index=False)); nd.to_csv("negatives.csv", index=False)

    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    for r, n in enumerate(ims):
        raw, corr, m, blue = ims[n]; lakes = res[n][1]
        ax[r, 0].imshow(raw[..., ::-1]); ax[r, 0].set_title(f"{n}: оригінал")
        ax[r, 1].imshow(corr[..., ::-1]); ax[r, 1].set_title("CLAHE (канал L)")
        ax[r, 2].imshow(m, cmap="gray"); ax[r, 2].set_title("маска води")
        ov = raw.copy()

        for lake_part in lakes:
            cv2.drawContours(ov, [lake_part], -1, (0, 0, 255), 3)

        for oc in res[n][2][:6]: 
            cv2.drawContours(ov, [oc], -1, (0, 255, 255), 2)
            
        ax[r, 3].imshow(ov[..., ::-1]); ax[r, 3].set_title("контури")
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig("pipeline.png", dpi=80); plt.close()
    
    fig, a = plt.subplots(1, 2, figsize=(12, 6))
    for col, (n, cc) in zip(("r", "b"), (("Bing", cB), ("MapCarta", cM))):
        p = cc[:, 0, :].astype(float); M = cv2.moments(cc)
        p -= [M["m10"]/M["m00"], M["m01"]/M["m00"]]; p /= np.sqrt(cv2.contourArea(cc))
        a[0].plot(p[:, 0], -p[:, 1], col, label=n)
        a[1].plot(fourier_desc(cc), col, label=n)
    a[0].set_aspect("equal"); a[0].legend(); a[0].set_title("Контури (центр + масштаб)")
    a[1].set_title("Амплітуди дескрипторів Фур'є"); a[1].legend()
    plt.tight_layout(); plt.savefig("compare.png", dpi=80)

    fig, a = plt.subplots(1, 2, figsize=(12, 4))
    for ax_, n in zip(a, ims):
        raw, corr = ims[n][0], ims[n][1]
        for im_, lbl in ((raw, "до"), (corr, "після CLAHE")):
            ax_.hist(cv2.cvtColor(im_, cv2.COLOR_BGR2GRAY).ravel(), 64, alpha=.5, label=lbl, log=True)
        ax_.set_title(f"Гістограма яскравості: {n}"); ax_.legend()
    plt.tight_layout(); plt.savefig("hist.png", dpi=80); plt.close()

    raw = load(*SRC["Bing"]); base_parts = res["Bing"][1]; base = np.vstack(base_parts) if len(base_parts) > 1 else base_parts[0]
    vr = []
    rng = np.random.default_rng(0)
    def rot(im, a):
        h, w = im.shape[:2]; M = cv2.getRotationMatrix2D((w/2, h/2), a, 1)
        return cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    tests = {"поворот 30°": lambda im: rot(im, 30),
             "масштаб 0.6": lambda im: cv2.resize(im, None, fx=.6, fy=.6, interpolation=cv2.INTER_AREA),
             "шум σ=15": lambda im: np.clip(im + rng.normal(0, 15, im.shape), 0, 255).astype(np.uint8),
             "яскравість ×0.7": lambda im: np.clip(im * .7, 0, 255).astype(np.uint8)}
    for t, fn in tests.items():
        m, _ = water_mask(clahe_L(fn(raw))); lp, _ = main_lake(m)
        c = np.vstack(lp) if len(lp) > 1 else lp[0]
        d = compare(base, c); d["IoU_aligned"] = iou_aligned(base, c, None); d["test"] = t; vr.append(d)
    vd = pd.DataFrame(vr).round(4)[["test", "matchShapes", "fourier_L2", "IoU_aligned"]]
    print("\nВерифікація (Bing vs перетворений Bing):\n", vd.to_string(index=False)); vd.to_csv("verify.csv", index=False)