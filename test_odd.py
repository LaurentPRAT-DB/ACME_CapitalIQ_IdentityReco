import sys

def odd(number: int):
    if number < 5:
        print("Fail")
    elif number == 6 or number == 7:
        print("GG")
    elif number >= 5 and number <9:
        print("Pass")
    else:
        print("EXCELLENT")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        number = int(sys.argv[1])
        odd(number)
    else:
        print("Usage: python3 test_odd.py <number>")
        print("Example: python3 test_odd.py 5")
