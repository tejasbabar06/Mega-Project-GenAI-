import TaskItem from "./TaskItem";

/**
 * TaskList Component - Renders the list of tasks
 * Uses map() to iterate over tasks array
 * Props: tasks, onToggle, onDelete
 */
function TaskList({ tasks, onToggle, onDelete }) {
  // Show a message if no tasks exist
  if (tasks.length === 0) {
    return <p className="no-tasks">No tasks yet. Add one above!</p>;
  }

  return (
    <ul className="task-list">
      {/* Use map() to render each task, using task.id as key */}
      {tasks.map((task) => (
        <TaskItem
          key={task.id}
          task={task}
          onToggle={onToggle}
          onDelete={onDelete}
        />
      ))}
    </ul>
  );
}

export default TaskList;
