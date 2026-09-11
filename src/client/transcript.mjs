// Display hypotheses separately from committed, audibly completed conversation entries.
export class Transcript {
  constructor(container, caption, speaker, text) {
    Object.assign(this, { container, caption, speaker, text });
    this.entries = [];
    this.reset();
  }

  show(who, text, partial = false) {
    this.caption.dataset.speaker = who;
    this.caption.classList.toggle("partial", partial);
    this.caption.classList.remove("waiting");
    this.speaker.textContent =
      who === "You" ? (partial ? "You · listening" : "You") : "Zen";
    this.text.textContent = text;
  }

  append(who, text, generation) {
    if (!text.trim()) return;
    this.container.querySelector(".transcript-empty")?.remove();
    const last = this.entries.at(-1);
    if (
      who === "Zen" &&
      last?.who === who &&
      last.generation === generation &&
      last.body.textContent.length + text.length < 32768
    ) {
      last.body.textContent += ` ${text}`;
    } else {
      const row = document.createElement("p");
      row.className = "transcript-entry";
      row.dataset.speaker = who;
      const label = document.createElement("span");
      label.className = "speaker";
      label.textContent = who;
      const body = document.createElement("span");
      body.textContent = text;
      row.append(label, body);
      this.container.append(row);
      this.entries.push({ row, body, who, generation });
      while (this.entries.length > 40) this.entries.shift().row.remove();
    }
    this.container.scrollTop = this.container.scrollHeight;
  }

  waiting() {
    this.caption.classList.add("waiting");
  }

  reset() {
    this.entries = [];
    this.container.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "transcript-empty";
    empty.textContent = "A fresh page. Say hello.";
    this.container.append(empty);
    this.caption.classList.remove("partial", "waiting");
    delete this.caption.dataset.speaker;
    this.speaker.textContent = "Your conversation starts here";
    this.text.textContent = "What’s on your mind?";
  }
}
