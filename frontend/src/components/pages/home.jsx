import React, { useState } from "react";

const Home = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setResult("Please select a JPG file first.");
      return;
    }

    setLoading(true);
    setResult("");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      const data = await res.json();
      setResult(data.result);
    } catch (err) {
      setResult("Error: " + err.message);
    } finally {
      setLoading(false);
      setFile(null);
    }
  };

  return (
    <div className="flex flex-col items-center h-screen">
      <div className="w-full p-8 mb-4 bg-blue-500">
        <h1 className="text-5xl text-center text-white font-bold">
          IMAGE CAPTIONING
        </h1>
      </div>

      <form
        onSubmit={handleSubmit}
        className="flex flex-col items-center p-4 m-4"
      >
        <h2 className="text-2xl m-2">Upload File</h2>

        <div className="flex items-center">
          <input
            type="file"
            accept="image/jpeg, image/jpg"
            onChange={(e) => {
              setResult("");
              setFile(e.target.files[0]);
            }}
            className="border rounded p-2 block text-sm
                       file:cursor-pointer file:bg-blue-300
                       file:rounded-2xl file:p-2 file:font-semibold
                       file:mx-2 hover:file:bg-blue-400"
          />

          <button
            type="submit"
            disabled={loading}
            className="mx-4 p-2 border rounded w-20 text-xl
                       bg-green-300 font-semibold cursor-pointer
                       hover:bg-green-500 disabled:opacity-50
                       disabled:cursor-not-allowed"
          >
            {loading ? "Loading…" : "GO!"}
          </button>
        </div>
      </form>

      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-2">Result:</h2>
          <p className="text-gray-800">{result}</p>
        </div>
      )}
    </div>
  );
};

export default Home;
