# script_with_tests.py

def add(a: int, b: int) -> int:
    """A simple function to add two numbers."""
    return a + b

def test_add_positive_numbers():
    """Test case for adding positive numbers."""
    result = add(2, 3)
    assert result == 5

def test_add_zero():
    """Test case for adding zero."""
    result = add(5, 0)
    assert result == 5
    
    


if __name__ == "__main__":
    # This code runs only when the file is executed directly with 'python'
    print(f"Result of add(10, 20): {add(10, 20)}")
