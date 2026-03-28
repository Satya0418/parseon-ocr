const handwrittenInput = document.getElementById("handwrittenInput");
const handwrittenPreview = document.getElementById("handwrittenPreview");
const predictHandwrittenBtn = document.getElementById("predictHandwrittenBtn");

const handwrittenPrediction = document.getElementById("handwrittenPrediction");
const handwrittenExtracted = document.getElementById("handwrittenExtracted");
const handwrittenDetails = document.getElementById("handwrittenDetails");

const pdfInput = document.getElementById("pdfInput");
const predictPdfBtn = document.getElementById("predictPdfBtn");
const pdfShow = document.getElementById("pdfShow");
const pdfExtracted = document.getElementById("pdfExtracted");
const pdfDetails = document.getElementById("pdfDetails");
const pdfPageOutputs = document.getElementById("pdfPageOutputs");

let handwrittenFile = null;
let pdfFile = null;

handwrittenInput.addEventListener("change", () => {
  const file = handwrittenInput.files?.[0];
  handwrittenFile = file || null;

  if (!handwrittenFile) {
    handwrittenPreview.style.display = "none";
    predictHandwrittenBtn.disabled = true;
    return;
  }

  handwrittenPreview.src = URL.createObjectURL(handwrittenFile);
  handwrittenPreview.style.display = "block";
  predictHandwrittenBtn.disabled = false;
});

predictHandwrittenBtn.addEventListener("click", async () => {
  if (!handwrittenFile) return;

  predictHandwrittenBtn.disabled = true;
  predictHandwrittenBtn.textContent = "Extracting...";

  const formData = new FormData();
  formData.append("file", handwrittenFile);

  try {
    const resp = await fetch("/predict-handwritten", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Handwritten extraction failed");
    }

    const data = await resp.json();
    handwrittenPrediction.textContent = data.prediction || "(empty)";
    handwrittenExtracted.textContent = (data.prediction || "").trim() || "(empty extraction)";
    handwrittenDetails.textContent = `confidence=${(data.confidence * 100).toFixed(2)}%, timesteps=${data.timesteps}, classes=${data.num_classes}`;
  } catch (error) {
    handwrittenPrediction.textContent = "Error";
    handwrittenExtracted.textContent = "-";
    handwrittenDetails.textContent = error.message;
  } finally {
    predictHandwrittenBtn.disabled = false;
    predictHandwrittenBtn.textContent = "Extract Handwritten";
  }
});

pdfInput.addEventListener("change", () => {
  const file = pdfInput.files?.[0];
  pdfFile = file || null;
  predictPdfBtn.disabled = !pdfFile;
});

predictPdfBtn.addEventListener("click", async () => {
  if (!pdfFile) return;

  predictPdfBtn.disabled = true;
  predictPdfBtn.textContent = "Extracting PDF...";

  const formData = new FormData();
  formData.append("file", pdfFile);

  try {
    const resp = await fetch("/predict-pdf", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "PDF extraction failed");
    }

    const data = await resp.json();
    pdfShow.textContent = `${data.pages} page(s) processed`;
    pdfExtracted.textContent = (data.extracted_text || "").trim() || "(empty extraction)";
    pdfDetails.textContent = data.page_results
      .map((x) => `p${x.page}:${(x.confidence * 100).toFixed(1)}%`)
      .join(" | ");
    pdfPageOutputs.textContent = JSON.stringify(data.page_results, null, 2);
  } catch (error) {
    pdfShow.textContent = "Error";
    pdfExtracted.textContent = "-";
    pdfDetails.textContent = error.message;
    pdfPageOutputs.textContent = "-";
  } finally {
    predictPdfBtn.disabled = false;
    predictPdfBtn.textContent = "Extract PDF";
  }
});
