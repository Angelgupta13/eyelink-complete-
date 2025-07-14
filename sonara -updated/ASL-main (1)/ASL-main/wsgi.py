from app import app

if __name__ == "__main__":
    # This part is optional, but can be useful if you ever run wsgi.py directly
    # It will default to Flask's dev server, not Waitress, unless you add Waitress here too.
    # For Gunicorn, Gunicorn itself will import 'app' from this file.
    print("To run with Gunicorn (on server): gunicorn --workers 4 --bind 0.0.0.0:5000 wsgi:app")
    print("To run locally with Waitress: python app.py")
    # app.run(debug=True, host='0.0.0.0', port=5000) # Example if running wsgi.py directly