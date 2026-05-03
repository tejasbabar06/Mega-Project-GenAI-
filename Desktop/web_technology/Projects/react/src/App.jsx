import { useState } from "react";
import TaskInput from "./components/TaskInput";
import TaskList from "./components/TaskList";
import "./App.css";

/**
 * App Component - Main component that manages all task state
 * Uses useState hook to store the array of task objects
 */
function App() {
  // State to hold all tasks
  const [tasks, setTasks] = useState([]);

  // Add a new task to the list
  const addTask = (text) => {
    const newTask = {
      id: Date.now(), // unique id using timestamp
      text: text,
      completed: false,
    };
    setTasks([...tasks, newTask]); // spread existing tasks and add new one
  };

  // Toggle the completed status of a task
  const toggleComplete = (id) => {
    setTasks(
      tasks.map((task) =>
        task.id === id ? { ...task, completed: !task.completed } : task
      )
    );
  };

  // Delete a task by filtering it out
  const deleteTask = (id) => {
    setTasks(tasks.filter((task) => task.id !== id));
  };

  // Calculate completed task count
  const completedCount = tasks.filter((task) => task.completed).length;

  return (
    <div className="app-container">
      <h1 className="app-title">📝 Task Manager</h1>

      {/* Input component for adding new tasks */}
      <TaskInput onAdd={addTask} />

      {/* Display total and completed task counts */}
      <div className="task-stats">
        <span>Total Tasks: <strong>{tasks.length}</strong></span>
        <span>Completed: <strong>{completedCount}</strong></span>
      </div>

      {/* List component to display all tasks */}
      <TaskList
        tasks={tasks}
        onToggle={toggleComplete}
        onDelete={deleteTask}
      />
    </div>
  );
}

export default App;
