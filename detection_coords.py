"""
detection_coords.py
-------------------
Utilities for extracting center (x, y) coordinates from SAM3 detection results.

The centroid is computed server-side as the mean of all mask pixel positions,
so it lands at the visual center of the object regardless of its shape.
For long objects (screwdriver, saw, tape measure) this gives the midpoint
of the span rather than a corner or edge.

Usage
-----
    from detection_coords import get_center, get_all_centers

    result = model.detect.remote(tools, img_bytes, descriptions)

    # Single-tool shortcut
    cx, cy = get_center(result, "hammer")  # returns (x, y) or None

    # All tools at once
    for entry in get_all_centers(result):
        print(entry["label"], entry["cx"], entry["cy"])
"""


def get_center(result: dict, label: str = None) -> tuple[int, int] | None:
    """
    Return the (x, y) center of a detected object.

    Parameters
    ----------
    result : dict
        The dict returned by DetectTools.detect.remote(). Must contain a
        "detections" list where each item has "label", "cx", and "cy".
    label : str, optional
        If given, return the center for the first detection matching this
        label (case-insensitive). If None, return the center of the first
        detection regardless of label.

    Returns
    -------
    (cx, cy) tuple of ints, or None if no matching detection is found.
    """
    for d in result.get("detections", []):
        if label is None or d.get("label", "").lower() == label.lower():
            cx = d.get("cx")
            cy = d.get("cy")
            if cx is not None and cy is not None:
                return (int(cx), int(cy))
    return None


def get_all_centers(result: dict) -> list[dict]:
    """
    Return the center coordinates for every detected object.

    Parameters
    ----------
    result : dict
        The dict returned by DetectTools.detect.remote().

    Returns
    -------
    List of dicts, each with keys:
        "label" : str   – tool name
        "cx"    : int   – x coordinate of center (pixels from left)
        "cy"    : int   – y coordinate of center (pixels from top)
        "score" : float – detection confidence score
    Only detections that include coordinate data are included.
    """
    centers = []
    for d in result.get("detections", []):
        cx = d.get("cx")
        cy = d.get("cy")
        if cx is not None and cy is not None:
            centers.append({
                "label": d.get("label", "unknown"),
                "cx":    int(cx),
                "cy":    int(cy),
                "score": d.get("score", 0.0),
            })
    return centers
