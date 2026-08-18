#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import numpy as np

from edit_polygon_label import create_editor_window


class EditorWindowTests(unittest.TestCase):
    def test_frame_is_presented_before_mouse_callback(self):
        order = []
        preview = np.zeros((40, 60, 3), np.uint8)
        with patch("edit_polygon_label.cv2.namedWindow", side_effect=lambda *_: order.append("named")), \
             patch("edit_polygon_label.cv2.imshow", side_effect=lambda *_: order.append("shown")), \
             patch("edit_polygon_label.cv2.waitKey", side_effect=lambda *_: order.append("waited") or -1), \
             patch("edit_polygon_label.cv2.resizeWindow", side_effect=lambda *_: order.append("resized")), \
             patch("edit_polygon_label.cv2.setMouseCallback", side_effect=lambda *_: order.append("callback")):
            create_editor_window("editor", preview, lambda *_: None)
        self.assertLess(order.index("shown"), order.index("callback"))
        self.assertLess(order.index("waited"), order.index("callback"))


if __name__ == "__main__":
    unittest.main()
