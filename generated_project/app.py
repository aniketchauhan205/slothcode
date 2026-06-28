from flask import Flask, g, render_template, request, jsonify
import sqlite3

app = Flask(__name__)
app.config['DATABASE'] = 'tasks.db'

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'],
            detect_threads=False # Allow multiple threads to use the same connection
        )
        g.db.row_factory = sqlite3.Row
    return g.db

def close_connection(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )
        db.commit()

def add_task(task_text):
    with app.app_context():
        db = get_db()
        db.execute("INSERT INTO tasks (task, status) VALUES (?, ?)", (task_text, 'pending'))
        db.commit()
        # Return the id of the newly inserted task
        cursor = db.execute("SELECT last_insert_rowid()")
        task_id = cursor.fetchone()[0]
        return task_id

def get_all_tasks():
    with app.app_context():
        db = get_db()
        cursor = db.execute("SELECT id, task, status FROM tasks")
        tasks = []
        for row in cursor.fetchall():
            tasks.append(dict(row))
        return tasks

def update_task_status(task_id, new_status):
    with app.app_context():
        db = get_db()
        db.execute("UPDATE tasks SET status = ? WHERE id = ?", (new_status, task_id))
        db.commit()

def delete_task_by_id(task_id):
    with app.app_context():
        db = get_db()
        db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        db.commit()

@app.cli.command('init-db')
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    print('Initialized the database.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tasks', methods=['POST'])
def create_task():
    data = request.get_json()
    task_text = data.get('task')
    if not task_text:
        return jsonify({'error': 'Task description is required'}), 400
    task_id = add_task(task_text)
    return jsonify({'message': 'Task added successfully', 'id': task_id}), 201

@app.route('/tasks', methods=['GET'])
def get_tasks():
    tasks = get_all_tasks()
    return jsonify(tasks)

@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    data = request.get_json()
    new_status = data.get('status')
    if not new_status:
        return jsonify({'error': 'Status is required'}), 400
    update_task_status(task_id, new_status)
    return jsonify({'message': f'Task {task_id} status updated to {new_status}'})

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    delete_task_by_id(task_id)
    return jsonify({'message': f'Task {task_id} deleted successfully'})

app.teardown_appcontext(close_connection)

if __name__ == '__main__':
    app.run(debug=True)
