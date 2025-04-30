// Base URL for your FastAPI backend
const BASE_URL = "http://localhost:8000";

export async function generateImageFromPrompt(promptText) {
  const res = await fetch(`${BASE_URL}/generate-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: promptText }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || res.statusText);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export function previewLocalImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed reading file"));
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });
}

export async function generateCaptionFromFile(file) {
  const form = new FormData();
  form.append("image", file);

  const res = await fetch(`${BASE_URL}/caption`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || res.statusText);
  }

  const { caption } = await res.json();
  return caption;
}
