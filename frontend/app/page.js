import React, { useState } from "react";
import SearchBar from "../components/SearchBar";
import MyEngineResults from "../components/MyEngineResults";
import GoogleResults from "../components/GoogleResults";
import BingResults from "../components/BingResults";

export default function Home() {
  const [engineResults, setEngineResults] = useState([]);
  const [googleResults, setGoogleResults] = useState([]);
  const [bingResults, setBingResults] = useState([]);

  const handleEngineSearch = async (query) => {
    setEngineResults(query);
    // For now, we'll just set empty lists for Google and Bing results
    setGoogleResults([]);
    setBingResults([]);
  };

  return (
    <main className="grid grid-cols-2 grid-rows-2 h-screen">
      {/* My Engine Frame  */}
      <div className="border border-gray-300 p-4">
        <h2 className="text-xl font-bold mb-4">
          My Engine
        </h2>
        <SearchBar onSearch={handleEngineSearch} />
        <MyEngineResults results={engineResults} />
      </div>

      {/* Google Frame */}
      <div className="border border-gray-300 p-4">
        <h2 className="text-xl font-bold mb-4">Google</h2>
      </div>


      {/* Bing Frame */}
      <div className="border border-gray-300 p-4">
        <h2 className="text-xl font-bold mb-4">Bing</h2>
      </div>

      {/* Clustering Frame */}
      <div className="border border-gray-300 p-4">
        <h2 className="text-xl font-bold mb-4">Clustering</h2>
      </div>
    </main>
  );
}