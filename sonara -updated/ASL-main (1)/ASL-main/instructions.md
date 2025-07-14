0. clone and cd to the project root
1. make a new virtual environment: python -m venv .venv (once)
2. open the venv: .venv/Scripts/activate
3. install things: pip install -r requirements.txt
4. python app.py
(better alternative on linux-env: gunicorn --workers 4 --bind 0.0.0.0:5000 wsgi:app")
5. go to given link (deafult - [localhost 5000](http://127.0.0.1:5000))