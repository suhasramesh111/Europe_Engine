import React from 'react';

const GoogleResults = ({ results }) => {
  return (
    <div>
      {results.length > 0 ? (
        <ul>
          {results.map((result, index) => (
            <li key={index} className="mb-4">
              <a href={result.link} target="_blank" rel="noopener noreferrer" className="text-blue-500 underline">
                {result.link}
              </a>
              <p className="text-gray-700">{result.description}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p>No results found from Google.</p>
      )}
    </div>
  );
};

export default GoogleResults;