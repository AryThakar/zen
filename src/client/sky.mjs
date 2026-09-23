// The sky behind the page: three layers of stars, drawn once each and panned by the stylesheet at
// different speeds - the far ones slowest - so it has depth.
//
// Each layer is a canvas twice the window's width with its pattern drawn twice, side by side, so a
// pan by half its width comes back to where it started without a seam. A fixed seed keeps the sky
// the same every launch. Nothing is drawn per frame: a layer is redrawn only when the window
// changes size or the theme changes. Night: faint stars, a few bright ones with a soft glow, some a
// little cool or warm, larger and brighter the nearer the layer. Day: soft motes of light with a
// lavender or sky halo. Only the nearest layer is drawn at the screen's full sharpness; the far
// ones are specks, and drawing them at 1x keeps their memory small.

const DEPTHS = [
  { per: 3400, size: [0.35, 0.75], alpha: [0.12, 0.4], bright: 0.004, sharp: 1 },
  { per: 6500, size: [0.55, 1.05], alpha: [0.25, 0.55], bright: 0.02, sharp: 1 },
  { per: 14000, size: [0.9, 1.6], alpha: [0.45, 0.85], bright: 0.12, sharp: 1.5 },
];

const daylight = () => {
  const theme = document.documentElement.dataset.theme;
  return theme ? theme === "light" : matchMedia("(prefers-color-scheme: light)").matches;
};

function draw(canvases) {
  const w = innerWidth,
    h = innerHeight,
    light = daylight();
  canvases.forEach((canvas, depth) => {
    const d = DEPTHS[depth],
      scale = Math.min(d.sharp, devicePixelRatio || 1);
    canvas.width = Math.round(w * 2 * scale);
    canvas.height = Math.round(h * scale);
    const g = canvas.getContext("2d");
    g.setTransform(scale, 0, 0, scale, 0, 0);
    g.clearRect(0, 0, w * 2, h);
    let seed = 20260918 + depth * 7919;
    const rand = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
    const count = Math.round(((w * h) / d.per) * (light ? 0.55 : 1));
    for (let i = 0; i < count; i++) {
      const x = rand() * w,
        y = rand() * h,
        bright = rand() < d.bright,
        tint = rand();
      let size = d.size[0] + rand() * (d.size[1] - d.size[0]);
      let alpha = d.alpha[0] + rand() * (d.alpha[1] - d.alpha[0]);
      if (bright) {
        size *= 1.5;
        alpha = Math.min(1, alpha + 0.3);
      }
      const color = light ? "255,255,255" : tint < 0.22 ? "200,215,255" : tint > 0.9 ? "255,226,206" : "236,236,255";
      const halo = light ? (tint < 0.5 ? "140,120,255" : "110,175,255") : color;
      for (const at of [x, x + w]) {
        if (bright || light) {
          const r = size * (light ? 5.5 : 4.5);
          const glow = g.createRadialGradient(at, y, 0, at, y, r);
          glow.addColorStop(0, `rgba(${halo},${alpha * (light ? 0.42 : 0.3)})`);
          glow.addColorStop(1, `rgba(${halo},0)`);
          g.fillStyle = glow;
          g.beginPath();
          g.arc(at, y, r, 0, Math.PI * 2);
          g.fill();
        }
        g.fillStyle = `rgba(${color},${Math.min(1, alpha * (light ? 1.4 : 1))})`;
        g.beginPath();
        g.arc(at, y, size, 0, Math.PI * 2);
        g.fill();
      }
    }
  });
}

export function startSky(root = document.querySelector(".space")) {
  const canvases = [...(root?.querySelectorAll(".layer canvas") || [])];
  if (!canvases.length) return;
  // Dragging the window's edge fires resize on every step; one redraw a frame is plenty.
  let queued = 0;
  const redraw = () => {
    cancelAnimationFrame(queued);
    queued = requestAnimationFrame(() => draw(canvases));
  };
  draw(canvases);
  addEventListener("resize", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", redraw);
}

if (typeof document !== "undefined") startSky();
