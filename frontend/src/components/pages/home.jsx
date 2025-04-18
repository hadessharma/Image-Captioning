import React from "react";
import { useState } from "react";

const Home = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("STRING");
  const handleSubmit = (e) => {
    e.preventDefault();
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
        <div className="flex">
          <input
            type="file"
            accept="image/jpeg, image/jpg"
            onChange={(e) => setFile(e.target.files[0])}
            className="border-1 rounded p-2 block text-sm file:cursor-pointer file:bg-blue-300 file:rounded-2xl file:p-2 file:font-semibold file:mx-2 hover:file:bg-blue-400"
          />
          <button
            type="submit"
            className="mx-4 p-2 border-1 rounded w-20 text-xl bg-green-300 font-semibold cursor-pointer hover:bg-green-500"
          >
            GO!
          </button>
        </div>
      </form>
      {result && (
        <div>
          <h2>{result}</h2>
        </div>
      )}
    </div>
  );
};

export default Home;
