/**
 * TaskItem Component - Displays a single task
 * Shows task text with strike-through if completed
 * Props: task, onToggle, onDelete
 */
function TaskItem({ task, onToggle, onDelete }) {
  return (
    <li className={`task-item ${task.completed ? "completed" : ""}`}>
      {/* Task text - strike-through style applied via CSS when completed */}
      <span className="task-text">{task.text}</span>

      <div className="task-actions">
        {/* Toggle complete/incomplete */}
        <button
          className="btn btn-complete"
          onClick={() => onToggle(task.id)}
        >
          {task.completed ? "Undo" : "Complete"}
        </button>

        {/* Delete the task */}
        <button
          className="btn btn-delete"
          onClick={() => onDelete(task.id)}
        >
          Delete
        </button>
      </div>
    </li>
  );
}

export default TaskItem;
