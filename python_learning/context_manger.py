from contextlib import contextmanager


@contextmanager
def print_context():
    print("this is the begining of with")
    yield
    print("this is the end of with")


if __name__ == "__main__":
    with print_context():
        print("in the with context")
