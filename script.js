const form = document.getElementById("predict-form");
const btn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const errorBox = document.getElementById("error-box");
const resultBox = document.getElementById("result");
const resultTitle = document.getElementById("result-title");
const resultConf = document.getElementById("result-conf");
const barsBox = document.getElementById("bars");
const navToggle = document.getElementById("nav-toggle");
const navLinks = document.getElementById("nav-links");

// Keys here match the "key" values in train_model.py FEATURES and the input names in index.html.
const SAMPLES = {
  1: { alcohol: 14.23, malic_acid: 1.71, ash: 2.43, alcalinity_of_ash: 15.6, magnesium: 127, total_phenols: 2.8, flavanoids: 3.06, nonflavanoid_phenols: 0.28, proanthocyanins: 2.29, color_intensity: 5.64, hue: 1.04, od280: 3.92, proline: 1065 },
  2: { alcohol: 12.37, malic_acid: 0.94, ash: 1.36, alcalinity_of_ash: 10.6, magnesium: 88, total_phenols: 1.98, flavanoids: 0.57, nonflavanoid_phenols: 0.28, proanthocyanins: 0.42, color_intensity: 1.95, hue: 1.05, od280: 1.82, proline: 520 },
  3: { alcohol: 12.86, malic_acid: 1.35, ash: 2.32, alcalinity_of_ash: 18, magnesium: 122, total_phenols: 1.51, flavanoids: 1.25, nonflavanoid_phenols: 0.21, proanthocyanins: 0.94, color_intensity: 4.1, hue: 0.76, od280: 1.29, proline: 630 },
};

navToggle.addEventListener("click", () => navLinks.classList.toggle("open"));
navLinks.querySelectorAll("a").forEach((a) =>
  a.addEventListener("click", () => navLinks.classList.remove("open"))
);

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.add("show");
}

function clearError() {
  errorBox.classList.remove("show");
  errorBox.textContent = "";
  form.querySelectorAll("input").forEach((i) => i.classList.remove("invalid"));
}

function setLoading(isLoading) {
  btn.disabled = isLoading;
  btn.classList.toggle("loading", isLoading);
  btn.querySelector(".btn-text").textContent = isLoading ? "Predicting..." : "Predict Cultivar";
}

document.querySelectorAll("[data-sample]").forEach((chip) => {
  chip.addEventListener("click", () => {
    const sample = SAMPLES[chip.dataset.sample];
    Object.entries(sample).forEach(([key, value]) => {
      const input = document.getElementById(key);
      if (input) input.value = value;
    });
    clearError();
  });
});

resetBtn.addEventListener("click", () => {
  form.reset();
  clearError();
  resultBox.classList.remove("show");
});

function renderResult(data) {
  resultTitle.textContent = data.prediction;
  resultConf.textContent = data.confidence;
  barsBox.innerHTML = "";

  data.probabilities.forEach((p) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const isTop = p.label === data.prediction;
    row.innerHTML =
      '<span>' + p.label + '</span>' +
      '<div class="bar-track"><div class="bar-fill' + (isTop ? " top" : "") + '"></div></div>' +
      '<span class="bar-val">' + p.value + '%</span>';
    barsBox.appendChild(row);
    // Animate the bar width after it is in the DOM.
    requestAnimationFrame(() => {
      setTimeout(() => { row.querySelector(".bar-fill").style.width = p.value + "%"; }, 50);
    });
  });

  resultBox.classList.add("show");
  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  resultBox.classList.remove("show");

  const payload = {};
  let missing = false;
  form.querySelectorAll("input").forEach((input) => {
    if (input.value.trim() === "") {
      input.classList.add("invalid");
      missing = true;
    }
    payload[input.name] = input.value;
  });

  if (missing) {
    showError("Please fill in all 13 fields, or use one of the quick samples.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      showError(data.error || "Prediction failed.");
    } else {
      renderResult(data);
    }
  } catch (err) {
    showError("Could not reach the server. Make sure app.py is running.");
  } finally {
    setLoading(false);
  }
});
