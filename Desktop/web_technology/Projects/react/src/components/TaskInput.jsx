import { useState } from "react";

/**
 * TaskInput Component - Handles the input field and Add button
 * Props: onAdd - function to add a new task
 */
function TaskInput({ onAdd }) {
  // State for the input field value
  const [inputValue, setInputValue] = useState("");

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault(); // prevent page reload

    // Do not allow empty tasks (trim removes whitespace)
    if (inputValue.trim() === "") {
      alert("Please enter a task!");
      return;
    }

    onAdd(inputValue.trim()); // pass the task text to parent
    setInputValue(""); // clear input after adding
  };

  return (
    <form className="task-input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        className="task-input"
        placeholder="Enter a new task..."
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
      />
      <button type="submit" className="btn btn-add">
        Add Task
      </button>
    </form>
  );
}

export default TaskInput;
