import React from 'react';

const BingResults = ({ results }) => {
  return (
    <div>
      {results.length > 0 ? (
        <ul>
          {results.map((result, index) => (
            <li key={index} className="mb-2">
              <a href={result.url} target="_blank" rel="noopener noreferrer" className="text-blue-500">
                {result.name}
              </a>
              <p className="text-gray-600">{result.snippet}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p>No results found.</p>
      )}
    </div>
  );
};

export default BingResults;