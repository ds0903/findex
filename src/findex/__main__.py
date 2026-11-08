from findex.cli import app

# без цієї перевірки `python -m findex index --executor processes` запускав би CLI ще раз
# в кожному дочірньому процесі: spawn заново імпортує головний модуль
if __name__ == "__main__":
    app()
