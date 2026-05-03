import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

// Render the App component into the root div
createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
