"""
Generate Picture Studio app icon and logo.
Run: python assets/generate_icon.py
"""
import math
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Colour palette ─────────────────────────────────────────────────────────────
BG_DARK   = (13,  17,  23)       # #0D1117  GitHub-dark
BG_CARD   = (22,  27,  34)       # #161B22
BLUE_1    = (31,  111, 235)      # #1F6FEB  primary
BLUE_2    = (56,  139, 253)      # #388BFD  lighter
TEAL      = (57,  197, 207)      # #39C5CF  accent
GREEN     = (63,  185, 80)       # #3FB950  success / bbox
PURPLE    = (188, 140, 255)      # #BC8CFF  ROI
ORANGE    = (210, 153, 34)       # #D29922  warning
WHITE     = (255, 255, 255)
GREY      = (139, 148, 158)      # #8B949E


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def rounded_rect_mask(size, radius):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size[0]-1, size[1]-1], radius=radius, fill=255)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ICON  (1024 × 1024)
# ══════════════════════════════════════════════════════════════════════════════

def draw_icon(size=1024) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # ── Background ──────────────────────────────────────────────────────────
    radius = int(s * 0.22)
    mask   = rounded_rect_mask((s, s), radius)

    bg = Image.new("RGBA", (s, s))
    bg_d = ImageDraw.Draw(bg)
    for y in range(s):
        t = y / s
        c = lerp_color((16, 24, 40), (8, 12, 20), t)
        bg_d.line([(0, y), (s, y)], fill=c + (255,))
    img.paste(bg, mask=mask)

    # ── Radial glow ─────────────────────────────────────────────────────────
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    cx, cy = s // 2, int(s * 0.46)
    for r in range(int(s * 0.55), 0, -1):
        alpha = int(32 * (1 - r / (s * 0.55)))
        gd.ellipse([cx-r, cy-r, cx+r, cy+r], fill=BLUE_1 + (alpha,))
    img = Image.alpha_composite(img, glow)
    d   = ImageDraw.Draw(img)

    # ── Outer ring (camera lens) ─────────────────────────────────────────────
    ring_w  = int(s * 0.048)
    r_ring  = int(s * 0.385)
    # ring glow layers
    for i in range(8):
        rr = r_ring + i * 2
        a  = int(60 * (1 - i / 8))
        d.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], outline=BLUE_2 + (a,), width=1)
    # solid ring
    for i in range(ring_w):
        t = i / ring_w
        col = lerp_color(BLUE_2, BLUE_1, t) + (255,)
        rr  = r_ring - i
        d.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], outline=col, width=1)

    # ── Aperture blades (clearly visible) ──────────────────────────────────
    r_inner = r_ring - ring_w
    blade_outer = int(r_inner * 0.96)
    blade_inner = int(r_inner * 0.28)
    for k in range(6):
        a0 = math.radians(k * 60 + 15)
        a1 = math.radians(k * 60 - 15)
        a_mid = math.radians(k * 60 + 180)
        tip_x = cx + math.cos(a_mid) * blade_inner
        tip_y = cy + math.sin(a_mid) * blade_inner
        e1_x  = cx + math.cos(a0) * blade_outer
        e1_y  = cy + math.sin(a0) * blade_outer
        e2_x  = cx + math.cos(a1) * blade_outer
        e2_y  = cy + math.sin(a1) * blade_outer
        blade_col = lerp_color(BLUE_1, (8, 14, 28), 0.45) + (230,)
        d.polygon([(tip_x, tip_y), (e1_x, e1_y), (e2_x, e2_y)], fill=blade_col)

    # ── Lens glass (inner circle) ────────────────────────────────────────────
    rg = int(r_inner * 0.97)
    d.ellipse([cx-rg, cy-rg, cx+rg, cy+rg], fill=(10, 16, 30, 245))

    # ── Inner scene: simulated image with label boxes ────────────────────────
    scene_r = int(r_inner * 0.88)

    # "Image" background gradient inside lens
    scene_img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scene_img)
    for y in range(cy - scene_r, cy + scene_r):
        t = (y - (cy - scene_r)) / (2 * scene_r)
        c = lerp_color((14, 22, 42), (18, 28, 52), t)
        # clip to circle
        dx2 = scene_r**2 - (y - cy)**2
        if dx2 < 0: continue
        dx = int(dx2**0.5)
        sd.line([(cx - dx, y), (cx + dx, y)], fill=c + (200,))
    img = Image.alpha_composite(img, scene_img)
    d = ImageDraw.Draw(img)

    # ── Bounding boxes INSIDE lens ───────────────────────────────────────────
    lw = max(2, int(s * 0.007))

    # Green box — detection result, upper portion
    b1_pad = int(scene_r * 0.08)
    box1 = [
        cx + int(scene_r * 0.02),  cy - int(scene_r * 0.72),
        cx + int(scene_r * 0.78),  cy + int(scene_r * 0.10),
    ]
    d.rectangle(box1, outline=GREEN + (230,), width=lw)
    tick = int(s * 0.032)
    for bx, by, ddx, ddy in [
        (box1[0], box1[1], 1, 1), (box1[2], box1[1], -1, 1),
        (box1[0], box1[3], 1, -1), (box1[2], box1[3], -1, -1),
    ]:
        d.line([(bx, by), (bx + ddx*tick, by)], fill=GREEN+(255,), width=lw+1)
        d.line([(bx, by), (bx, by + ddy*tick)], fill=GREEN+(255,), width=lw+1)

    # Purple ROI box — lower-left
    box2 = [
        cx - int(scene_r * 0.78), cy - int(scene_r * 0.15),
        cx + int(scene_r * 0.05), cy + int(scene_r * 0.72),
    ]
    d.rectangle(box2, outline=PURPLE + (210,), width=lw)
    tick2 = int(s * 0.024)
    for bx, by, ddx, ddy in [
        (box2[0], box2[1], 1, 1), (box2[2], box2[1], -1, 1),
        (box2[0], box2[3], 1, -1), (box2[2], box2[3], -1, -1),
    ]:
        d.line([(bx, by), (bx + ddx*tick2, by)], fill=PURPLE+(255,), width=lw)
        d.line([(bx, by), (bx, by + ddy*tick2)], fill=PURPLE+(255,), width=lw)

    # ── Lens glint (white arc, upper-left) ──────────────────────────────────
    gr = int(r_inner * 0.72)
    d.arc([cx-gr, cy-gr, cx+gr, cy+gr], start=215, end=255,
          fill=(255, 255, 255, 100), width=int(s * 0.013))

    # ── Centre focus dot ────────────────────────────────────────────────────
    fd = int(s * 0.038)
    for i in range(5):
        r2 = fd - i * 2
        a  = int(220 * (1 - i/5))
        if r2 > 0:
            d.ellipse([cx-r2, cy-r2, cx+r2, cy+r2], fill=BLUE_2 + (a,))

    # ── Neural network strip (bottom) ───────────────────────────────────────
    ny     = int(s * 0.905)
    nr     = int(s * 0.022)
    n_nd   = 5
    n_sp   = int(s * 0.145)
    nx0    = cx - n_sp * (n_nd - 1) // 2
    ncolors = [BLUE_1, TEAL, BLUE_2, TEAL, BLUE_1]
    for i in range(n_nd):
        nx = nx0 + i * n_sp
        if i < n_nd - 1:
            nx2 = nx0 + (i+1) * n_sp
            d.line([(nx, ny), (nx2, ny)],
                   fill=(56, 139, 253, 90), width=max(1, int(s*0.004)))
        # outer glow
        d.ellipse([nx-nr-3, ny-nr-3, nx+nr+3, ny+nr+3],
                  fill=ncolors[i] + (50,))
        d.ellipse([nx-nr, ny-nr, nx+nr, ny+nr], fill=ncolors[i] + (210,))

    # ── Apply rounded mask ──────────────────────────────────────────────────
    img.putalpha(mask)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# LOGO  (wide banner, 800 × 200)
# ══════════════════════════════════════════════════════════════════════════════

def draw_logo(width=1200, height=240) -> Image.Image:
    img = Image.new("RGBA", (width, height), BG_DARK + (255,))
    d   = ImageDraw.Draw(img)

    # subtle gradient background
    for y in range(height):
        t = y / height
        c = lerp_color((14, 20, 34), BG_DARK, t)
        d.line([(0, y), (width, y)], fill=c + (255,))
    d = ImageDraw.Draw(img)

    # icon on left
    icon_size = int(height * 0.74)
    icon = draw_icon(icon_size * 2).resize((icon_size, icon_size), Image.LANCZOS)
    pad  = (height - icon_size) // 2
    img.paste(icon, (pad, pad), icon)

    # fonts
    font_size_big = int(height * 0.34)
    font_size_tag = int(height * 0.165)
    font_size_sm  = int(height * 0.13)
    for bold_path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font_bold = ImageFont.truetype(bold_path, font_size_big)
            break
        except Exception:
            font_bold = ImageFont.load_default()
    for reg_path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font_reg = ImageFont.truetype(reg_path, font_size_tag)
            font_sm  = ImageFont.truetype(reg_path, font_size_sm)
            break
        except Exception:
            font_reg = ImageFont.load_default()
            font_sm  = ImageFont.load_default()

    text_x = pad + icon_size + int(height * 0.14)
    text_y = int(height * 0.10)

    # "Picture" white
    d.text((text_x, text_y), "Picture", font=font_bold, fill=WHITE + (255,))
    bb = d.textbbox((text_x, text_y), "Picture", font=font_bold)

    # "Studio" blue, immediately after Picture
    gap = int(height * 0.045)
    studio_x = bb[2] + gap
    d.text((studio_x, text_y), "Studio", font=font_bold, fill=BLUE_2 + (255,))

    # version badge – right-aligned, same row as text, top-right area
    v_text = "v1.0.0"
    v_bb   = d.textbbox((0, 0), v_text, font=font_sm)
    v_w    = v_bb[2] - v_bb[0] + 18
    v_h    = v_bb[3] - v_bb[1] + 8
    v_x    = width - pad - v_w
    v_y    = text_y + 6
    d.rounded_rectangle(
        [v_x, v_y, v_x + v_w, v_y + v_h],
        radius=5, fill=(31, 111, 235, 70), outline=BLUE_1 + (140,), width=1
    )
    d.text((v_x + 9, v_y + 4), v_text, font=font_sm, fill=BLUE_2 + (230,))

    # tagline
    tag_y = bb[3] + int(height * 0.06)
    d.text((text_x, tag_y),
           "AI-powered Image Labeling & Analysis",
           font=font_reg, fill=GREY + (255,))

    # gradient accent line at bottom
    line_y  = height - int(height * 0.07)
    line_x0 = text_x
    line_x1 = width - pad
    for x in range(line_x0, line_x1):
        t = (x - line_x0) / max(line_x1 - line_x0, 1)
        col = lerp_color(BLUE_1, TEAL, t * 2) if t < 0.5 \
              else lerp_color(TEAL, BG_DARK, (t - 0.5) * 2)
        d.line([(x, line_y), (x, line_y + 2)], fill=col + (200,))

    return img


# ══════════════════════════════════════════════════════════════════════════════
# BUILD ICONSET
# ══════════════════════════════════════════════════════════════════════════════

def build_iconset(base_dir: str) -> None:
    iconset_dir = os.path.join(base_dir, "icon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512, 1024]
    master = draw_icon(1024)

    for sz in sizes:
        img = master.resize((sz, sz), Image.LANCZOS)
        img.save(os.path.join(iconset_dir, f"icon_{sz}x{sz}.png"))
        if sz <= 512:
            img2x = master.resize((sz * 2, sz * 2), Image.LANCZOS)
            img2x.save(os.path.join(iconset_dir, f"icon_{sz}x{sz}@2x.png"))

    print(f"Iconset written to {iconset_dir}")

    # Save 1024px standalone PNG
    master.save(os.path.join(base_dir, "icon_1024.png"))
    print("Saved icon_1024.png")

    # Logo banner
    logo = draw_logo(800, 200)
    logo.save(os.path.join(base_dir, "logo.png"))
    print("Saved logo.png")

    # Small toolbar icon (32px transparent)
    small = master.resize((32, 32), Image.LANCZOS)
    small.save(os.path.join(base_dir, "icon_32.png"))
    print("Saved icon_32.png")


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__))
    build_iconset(base)
    print("\nDone! Run to build .icns:")
    print("  iconutil -c icns assets/icon.iconset -o assets/icon.icns")
