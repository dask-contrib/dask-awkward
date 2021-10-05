from ..io import from_json


def json_data(type="numbers"):
    if type == "numbers":
        return from_json(_numbers())
    elif type == "records":
        return from_json(_records())


def _records():
    a1 = """
    [
        {
            "analysis": {
                "x": [1, 2, 3],
                "y": [],
                "z": [2],
                "t": [8, 8]
            }
        },
        {
            "analysis": {
                "x": [],
                "y": [2, 3, 8],
                "z": [4],
                "t": [3, 3, 3]
            }
        },
        {
            "analysis": {
                "x": [1],
                "y": [2, 3, 5],
                "z": [4, 4],
                "t": [3, 3, 1, 3]
            }
        }
    ]
    """

    a2 = """
    [
        {
            "analysis": {
                "x": [1],
                "y": [5, 2],
                "z": [],
                "t": [8, 8, 3]
            }
        },
        {
            "analysis": {
                "x": [1],
                "y": [],
                "z": [4, 3],
                "t": [3, 8]
            }
        },
        {
            "analysis": {
                "x": [1, 2, 3],
                "y": [2, 3, 5],
                "z": [4, 4, 13, 3, 1, 3, 5],
                "t": []
            }
        }
    ]
    """

    a3 = """
    [
        {
            "analysis": {
                "x": [],
                "y": [1],
                "z": [2, 1],
                "t": [8, 8, 1, 3]
            }
        },
        {
            "analysis": {
                "x": [2],
                "y": [2, 3],
                "z": [],
                "t": []
            }
        },
        {
            "analysis": {
                "x": [],
                "y": [],
                "z": [4, 4],
                "t": [3]
            }
        }
    ]
    """
    return [a1, a2, a3]


def _numbers():
    a1 = """
    [
      [
        [3, 2],
        [1],
        [5, 4],
        [6]
      ],
      [
        [7]
      ],
      [
        [8, 9],
        []
      ]
    ]
    """

    a2 = """
    [
      [
        [3, 1],
        [],
        [2, 4],
        [5]
      ],
      [
        [6]
      ],
      [
        [7, 8],
        [9]
      ]
    ]
    """

    a3 = """
    [
      [
        [3, 1],
        [],
        [2],
        [4, 5]
      ],
      [
        [6]
      ],
      [
        [7, 8],
        [9]
      ]
    ]
    """

    return [a1, a2, a3]
