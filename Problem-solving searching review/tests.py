from breadth_search import Map 
def test_input1():
    map = Map("inputs/input1.txt")
    assert "right, right, down, pick_up, left, left, down, leave" == map.run()
def test_input2():
    map = Map("inputs/input2.txt")
    assert "right, right, down, down, down, right, right, right, up, up, up, pick_up, left, down, down, down, left, left, left, left, down, leave" == map.run()
def test_input3():
    map = Map("inputs/input3.txt")
    assert "right, right, down, down, down, right, right, right, right, up, up, left, left, up, pick_up, right, right, down, down, down, left, left, left, left, up, left, left, down, down, leave" == map.run()
def test_input4():
    map = Map("inputs/input4.txt")
    assert "left, left, down, down, down, down, down, down, pick_up, up, up, up, up, up, right, right, down, down, down, down, down, leave" == map.run()
def test_input5():
    map = Map("inputs/input5.txt")
    assert "up, up, up, up, right, right, down, right, right, down, down, down, down, pick_up, up, up, up, up, up, right, right, down, down, down, down, down, leave" == map.run()

