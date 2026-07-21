document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const promptInput = document.getElementById("prompt-input");
  const maxTokensInput = document.getElementById("max-tokens");
  const temperatureInput = document.getElementById("temperature");
  const topKInput = document.getElementById("top-k");
  const topPInput = document.getElementById("top-p");

  const valMaxTokens = document.getElementById("val-max-tokens");
  const valTemperature = document.getElementById("val-temperature");
  const valTopK = document.getElementById("val-top-k");
  const valTopP = document.getElementById("val-top-p");

  const btnGenerate = document.getElementById("btn-generate");
  const outputBox = document.getElementById("output-box");
  const spinner = btnGenerate.querySelector(".spinner");
  const btnText = btnGenerate.querySelector(".btn-text");

  const badgeMode = document.getElementById("badge-mode");
  const badgeDevice = document.getElementById("badge-device");
  const badgeParams = document.getElementById("badge-params");

  // Sync sliders
  maxTokensInput.addEventListener("input", () => valMaxTokens.textContent = maxTokensInput.value);
  temperatureInput.addEventListener("input", () => valTemperature.textContent = temperatureInput.value);
  topKInput.addEventListener("input", () => valTopK.textContent = topKInput.value);
  topPInput.addEventListener("input", () => valTopP.textContent = topPInput.value);

  // Preset prompts
  document.querySelectorAll(".btn-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      promptInput.value = btn.getAttribute("data-prompt").replace(/\\n/g, "\n");
    });
  });

  // Fetch model metadata
  async function fetchModelInfo() {
    try {
      const res = await fetch("/api/info");
      if (res.ok) {
        const info = await res.json();
        badgeMode.textContent = info.model_mode || "diff_llama";
        badgeDevice.textContent = `Device: ${info.device || "cpu"}`;
        badgeParams.textContent = `Params: ${(info.params || 0).toLocaleString()}`;
      }
    } catch (e) {
      console.warn("Could not fetch model info:", e);
    }
  }

  fetchModelInfo();

  // Generate text handler
  btnGenerate.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    btnGenerate.disabled = true;
    spinner.classList.remove("hidden");
    btnText.textContent = "Generating...";
    outputBox.classList.remove("placeholder");

    try {
      const payload = {
        prompt: prompt,
        max_new_tokens: parseInt(maxTokensInput.value, 10),
        temperature: parseFloat(temperatureInput.value),
        top_k: parseInt(topKInput.value, 10),
        top_p: parseFloat(topPInput.value),
      };

      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      outputBox.textContent = data.generated_text;
    } catch (err) {
      outputBox.textContent = `Error generating text: ${err.message}`;
    } finally {
      btnGenerate.disabled = false;
      spinner.classList.add("hidden");
      btnText.textContent = "Generate Output";
    }
  });
});
