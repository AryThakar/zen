// The intro plays once, as the window opens: the opening of Zen's teaser, its mark and its sound.
// A click or a key goes straight to the page.
const intro = document.querySelector(".intro");
if (intro && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  let sound = null;
  // The sound keeps the intro's own clock. Decoding takes a moment, so it starts as far into the
  // clip as the animation already is, and the bells still ring as the moon rises.
  (async () => {
    try {
      const context = new AudioContext();
      const buffer = await context.decodeAudioData(await (await fetch("/startup.ogg")).arrayBuffer());
      await context.resume().catch(() => {});
      const [veil] = intro.getAnimations();
      const offset = (veil?.currentTime ?? Infinity) / 1000;
      if (!intro.isConnected || context.state !== "running" || !(offset < buffer.duration)) {
        await context.close();
        return;
      }
      const source = context.createBufferSource();
      const gain = context.createGain();
      source.buffer = buffer;
      source.connect(gain).connect(context.destination);
      source.onended = () => context.close();
      source.start(0, offset);
      sound = { context, gain };
    } catch {
      /* Silent is still an intro. */
    }
  })();
  const done = () => {
    intro.remove();
    removeEventListener("pointerdown", skip, true);
    removeEventListener("keydown", skip, true);
  };
  const skip = () => {
    document.body.classList.add("landed");
    // Faded rather than cut, so skipping does not click.
    if (sound) {
      sound.gain.gain.setTargetAtTime(0, sound.context.currentTime, 0.04);
      setTimeout(() => sound.context.close().catch(() => {}), 300);
    }
    done();
  };
  addEventListener("pointerdown", skip, true);
  addEventListener("keydown", skip, true);
  intro.addEventListener("animationend", (event) => {
    if (event.target === intro) done();
  });
} else intro?.remove();
