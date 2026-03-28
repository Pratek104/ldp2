import cv2
import numpy as np


VIDEO_SOURCE = "road_car_view.mp4"


class LaneDetector:
    def __init__(self, smoothing=0.25, max_missed_frames=8):
        self.smoothing = smoothing
        self.max_missed_frames = max_missed_frames
        self.previous_lines = {"left": None, "right": None}
        self.missed_frames = {"left": 0, "right": 0}

    def canny(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blur, 50, 150)

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        polygon = np.array(
            [[
                (int(width * 0.08), height),
                (int(width * 0.42), int(height * 0.62)),
                (int(width * 0.58), int(height * 0.62)),
                (int(width * 0.92), height),
            ]],
            dtype=np.int32,
        )

        mask = np.zeros_like(image)
        fill_color = 255 if image.ndim == 2 else (255,) * image.shape[2]
        cv2.fillPoly(mask, polygon, fill_color)
        return cv2.bitwise_and(image, mask)

    def detect_line_segments(self, cropped_edges):
        return cv2.HoughLinesP(
            cropped_edges,
            2,
            np.pi / 180,
            30,
            np.array([]),
            minLineLength=20,
            maxLineGap=120,
        )

    def classify_and_average(self, frame, lines):
        height, width = frame.shape[:2]
        y_bottom = height
        y_top = int(height * 0.62)
        lane_groups = {"left": [], "right": []}

        if lines is None:
            return {"left": None, "right": None}

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)

            if abs(dy) < 15 or length < 25:
                continue

            slope = dx / dy
            intercept = x1 - (slope * y1)
            bottom_x = slope * y_bottom + intercept

            if slope < -0.15 and bottom_x < width * 0.55:
                lane_groups["left"].append((slope, intercept, length))
            elif slope > 0.15 and bottom_x > width * 0.45:
                lane_groups["right"].append((slope, intercept, length))

        averaged_lines = {"left": None, "right": None}
        for lane_name, candidates in lane_groups.items():
            if not candidates:
                continue

            slopes = np.array([candidate[0] for candidate in candidates])
            intercepts = np.array([candidate[1] for candidate in candidates])
            weights = np.array([candidate[2] for candidate in candidates])

            avg_slope = np.average(slopes, weights=weights)
            avg_intercept = np.average(intercepts, weights=weights)
            averaged_lines[lane_name] = self.make_line_points(
                width,
                y_bottom,
                y_top,
                avg_slope,
                avg_intercept,
            )

        return averaged_lines

    def make_line_points(self, width, y_bottom, y_top, slope, intercept):
        x_bottom = int(np.clip((slope * y_bottom) + intercept, 0, width - 1))
        x_top = int(np.clip((slope * y_top) + intercept, 0, width - 1))
        return np.array([x_bottom, y_bottom, x_top, y_top], dtype=np.float32)

    def smooth_lines(self, detected_lines):
        smoothed_lines = {}

        for lane_name in self.previous_lines:
            current_line = detected_lines.get(lane_name)
            previous_line = self.previous_lines[lane_name]

            if current_line is None:
                self.missed_frames[lane_name] += 1
                if self.missed_frames[lane_name] > self.max_missed_frames:
                    self.previous_lines[lane_name] = None
                    smoothed_lines[lane_name] = None
                else:
                    smoothed_lines[lane_name] = previous_line
                continue

            self.missed_frames[lane_name] = 0
            if previous_line is None:
                blended_line = current_line
            else:
                blended_line = (
                    (1.0 - self.smoothing) * previous_line
                    + self.smoothing * current_line
                )

            self.previous_lines[lane_name] = blended_line
            smoothed_lines[lane_name] = blended_line

        return {
            lane_name: None if line is None else np.round(line).astype(np.int32)
            for lane_name, line in smoothed_lines.items()
        }

    def draw_lane_overlay(self, frame, lane_lines):
        overlay = np.zeros_like(frame)
        left_line = lane_lines.get("left")
        right_line = lane_lines.get("right")

        if left_line is not None and right_line is not None:
            lane_polygon = np.array(
                [[
                    (left_line[0], left_line[1]),
                    (left_line[2], left_line[3]),
                    (right_line[2], right_line[3]),
                    (right_line[0], right_line[1]),
                ]],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, lane_polygon, (40, 120, 40))

        if left_line is not None:
            cv2.line(
                overlay,
                (left_line[0], left_line[1]),
                (left_line[2], left_line[3]),
                (0, 255, 0),
                10,
            )

        if right_line is not None:
            cv2.line(
                overlay,
                (right_line[0], right_line[1]),
                (right_line[2], right_line[3]),
                (0, 255, 0),
                10,
            )

        return cv2.addWeighted(frame, 0.8, overlay, 1.0, 1.0)


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video source: {VIDEO_SOURCE}")

    detector = LaneDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        edges = detector.canny(frame)
        cropped_edges = detector.region_of_interest(edges)
        line_segments = detector.detect_line_segments(cropped_edges)
        lane_lines = detector.classify_and_average(frame, line_segments)
        smoothed_lines = detector.smooth_lines(lane_lines)
        output_frame = detector.draw_lane_overlay(frame, smoothed_lines)

        cv2.imshow("Real-Time Lane Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
