document.addEventListener('DOMContentLoaded', () => {
    fetchTasks();
    setupAddTask(); // Call function to set up add task functionality
});

async function fetchTasks() {
    try {
        const response = await fetch('/tasks');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const tasks = await response.json();
        renderTasks(tasks);
    } catch (error) {
        console.error('Error fetching tasks:', error);
    }
}

function renderTasks(tasks) {
    const taskList = document.getElementById('task-list');
    taskList.innerHTML = ''; // Clear existing content

    tasks.forEach(task => {
        const listItem = document.createElement('li');
        listItem.textContent = task.text;
        listItem.dataset.id = task.id;

        if (task.status === 'completed') {
            listItem.classList.add('completed');
        }

        const completeButton = document.createElement('button');
        completeButton.textContent = 'Complete';
        completeButton.classList.add('complete-button');
        completeButton.addEventListener('click', async () => {
            const taskId = listItem.dataset.id;
            try {
                const response = await fetch(`/tasks/${taskId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ status: 'completed' }),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                fetchTasks(); // Refresh task list
            } catch (error) {
                console.error('Error completing task:', error);
            }
        });

        const deleteButton = document.createElement('button');
        deleteButton.textContent = 'Delete';
        deleteButton.classList.add('delete-button');
        deleteButton.addEventListener('click', async () => {
            const taskId = listItem.dataset.id;
            try {
                const response = await fetch(`/tasks/${taskId}`, {
                    method: 'DELETE',
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                fetchTasks(); // Refresh task list
            } catch (error) {
                console.error('Error deleting task:', error);
            }
        });

        listItem.appendChild(completeButton);
        listItem.appendChild(deleteButton);
        taskList.appendChild(listItem);
    });
}

function setupAddTask() {
    const newTaskInput = document.getElementById('new-task-input');
    const addTaskButton = document.getElementById('add-task-button');

    const addTask = async () => {
        const taskText = newTaskInput.value.trim();
        if (taskText === '') {
            alert('Task cannot be empty!');
            return;
        }

        try {
            const response = await fetch('/tasks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ task: taskText }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            newTaskInput.value = ''; // Clear input field
            fetchTasks(); // Refresh task list
        } catch (error) {
            console.error('Error adding task:', error);
        }
    };

    addTaskButton.addEventListener('click', addTask);

    newTaskInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            addTask();
        }
    });
}