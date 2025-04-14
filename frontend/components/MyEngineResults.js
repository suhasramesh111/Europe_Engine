import React from 'react';

const MyEngineResults = ({ results }) => {
  return (
    <div>
      {results.length > 0 ? (
        <ul className="list-disc list-inside">
          {results.map((result, index) => (
            <li key={index} className="mb-2">
              <a
                href={result[0]}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:underline"
              >
                {result[0]}
              </a>
              <span className="ml-2 text-gray-600">
                (Score: {result[1].toFixed(3)})
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p>No results found.</p>
      )}
    </div>
  );
};

export default MyEngineResults;