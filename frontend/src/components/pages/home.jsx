import React, { useState } from "react";
import {
  generateImageFromPrompt,
  previewLocalImage,
  generateCaptionFromFile,
} from "../../functions/generate";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [localPreview, setLocalPreview] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [loadingGen, setLoadingGen] = useState(false);
  const [loadingCap, setLoadingCap] = useState(false);
  const [caption, setCaption] = useState("");
  const [capError, setCapError] = useState("");

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setCaption("");
    setCapError("");
    try {
      const dataUrl = await previewLocalImage(file);
      setLocalPreview(dataUrl);
    } catch (err) {
      console.error(err);
      alert("Could not preview image");
    }
  };

  const handleCaption = async () => {
    if (!selectedFile) return;
    setLoadingCap(true);
    setCaption("");
    setCapError("");
    try {
      const result = await generateCaptionFromFile(selectedFile);
      setCaption(result);
    } catch (err) {
      console.error(err);
      setCapError(err.message || "Error generating caption");
    } finally {
      setLoadingCap(false);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setLoadingGen(true);
    try {
      const imageUrl = await generateImageFromPrompt(prompt);
      setGeneratedImage(imageUrl);
    } catch (err) {
      console.error(err);
      alert("Failed to generate image");
    } finally {
      setLoadingGen(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center">
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow-lg p-8 space-y-10">
        {/* Direct Upload & Caption */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-800">
            Upload Your Image
          </h2>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500
                       file:mr-4 file:py-2 file:px-4
                       file:rounded-full file:border-0
                       file:text-sm file:font-medium
                       file:bg-blue-50 file:text-blue-700
                       hover:file:bg-blue-100"
          />

          {localPreview && (
            <div className="border rounded-lg overflow-hidden">
              <img
                src={localPreview}
                alt="preview"
                className="w-full object-contain"
              />
            </div>
          )}

          {localPreview && (
            <button
              onClick={handleCaption}
              disabled={loadingCap}
              className="mt-2 px-6 py-2 bg-green-600 text-white font-semibold rounded-lg
                         hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loadingCap ? "Captioning…" : "GO"}
            </button>
          )}

          {caption && (
            <p className="mt-4 text-gray-800">
              <strong>Caption:</strong> {caption}
            </p>
          )}
          {capError && (
            <p className="mt-4 text-red-600">
              <strong>Error:</strong> {capError}
            </p>
          )}
        </section>

        {/* Text-to-Image */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-800">
            Generate from Prompt
          </h2>
          <div className="flex space-x-2">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe your image..."
              className="flex-1 px-4 py-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
            <button
              onClick={handleGenerate}
              disabled={loadingGen}
              className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-r-lg
                         hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loadingGen ? "Generating…" : "Generate"}
            </button>
          </div>
          {generatedImage && (
            <div className="border rounded-lg overflow-hidden">
              <img
                src={generatedImage}
                alt="generated"
                className="w-full object-contain"
              />
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
