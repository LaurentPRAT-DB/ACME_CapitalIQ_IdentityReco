import sys

def odd(number: int):
    if number % 2:
        print("impair")
    else:
        print("pair")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        number = int(sys.argv[1])
        odd(number)
    else:
        print("Usage: python3 odd_test.py <number>")
        print("Example: python3 odd_test.py 5")
