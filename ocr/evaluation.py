"""ICDAR-style CER evaluation helpers for Greek squeeze OCR.

Adapted from the contest evaluation code cited in the original notebook:
Nicholas Howe, CC-BY 4.0, ICDAR 2026 Text Recognition on Greek Squeezes.
"""

from __future__ import annotations

import math
import os
import unicodedata
from os import listdir
from os.path import isfile, join

import numpy as np
from textdistance import levenshtein


class Box:
    """Coordinates for the four corners of one annotated character box."""

    def __init__(self, xnw, ynw, xsw, ysw, xse, yse, xne, yne):
        self.xnw = xnw
        self.ynw = ynw
        self.xsw = xsw
        self.ysw = ysw
        self.xse = xse
        self.yse = yse
        self.xne = xne
        self.yne = yne

    def bd(self):
        return [self.xnw, self.ynw, self.xsw, self.ysw, self.xse, self.yse, self.xne, self.yne]

    def center(self):
        return (
            (self.xnw + self.xne + self.xse + self.xsw) / 4,
            (self.ynw + self.yne + self.yse + self.ysw) / 4,
        )

    def radius(self):
        return (
            math.sqrt((self.xnw - self.xse) ** 2 + (self.ynw - self.yse) ** 2)
            + math.sqrt((self.xne - self.xsw) ** 2 + (self.yne - self.ysw) ** 2)
        ) / 4

    def area(self):
        return math.sqrt(
            ((self.xne - self.xnw) ** 2 + (self.yne - self.ynw) ** 2)
            * ((self.xsw - self.xnw) ** 2 + (self.ysw - self.ynw) ** 2)
        )

    def angle(self):
        return math.atan2(self.yne - self.ynw, self.xne - self.xnw)

    def dist(self, box):
        c1 = self.center()
        c2 = box.center()
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


GREEK_PROXY = {
    "Α": "A", "Β": "B", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z",
    "Η": "H", "Θ": "Q", "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M",
    "Ν": "N", "Ξ": "X", "Ο": "O", "Π": "P", "Ρ": "R", "Σ": "S",
    "Τ": "T", "Υ": "Y", "Φ": "F", "Χ": "C", "Ψ": "U", "Ω": "W",
}


def convert(stringin):
    """Convert Greek characters to uppercase Latin proxy characters."""
    decomposed = unicodedata.normalize("NFD", str(stringin))
    chars = []
    for char in decomposed:
        if unicodedata.combining(char):
            continue
        upper = char.upper()
        chars.append(GREEK_PROXY.get(upper, upper))
    return "".join(chars)


def readBoxFile(fname):
    """Read a contest letter-box annotation file."""
    angle = None
    boxes = []
    transcript = []
    lines = []

    try:
        with open(fname, "r", encoding="utf-8") as handle:
            phase = 0
            for count, line in enumerate(handle, start=1):
                line2 = line.replace("*", "").replace(" ", "")
                if line2 == "" or line2 == "\n":
                    continue
                if line[0] == "#":
                    phase += 1
                elif phase == 0:
                    print(f"Header problem in file {fname}.")
                elif phase == 1:
                    if angle is None:
                        angle = float(line)
                    else:
                        print(f"Multiple angle data found in {fname}.")
                elif phase == 2:
                    parts = line.split()
                    if len(parts) == 8:
                        box = Box(*[float(part) for part in parts])
                        if box.area() < 9:
                            print(f"Skipping box with very small area (line {count}).")
                        else:
                            boxes.append(box)
                    else:
                        print(f"Incorrect parameter count for box line ({len(parts)}) in {fname}.")
                elif phase == 3:
                    transcript.append(line.replace("*", "").replace("\n", "").replace("\r", "").replace(" ", ""))
                elif phase == 4:
                    parts = line.split()
                    if len(parts) == 4:
                        lines.append([float(part) for part in parts])
                        if np.any(np.isnan(lines[-1])):
                            print(f"Bad values in line specification (file line {count}) -- skipping.")
                            lines.pop()
                        elif lines[-1][0] == lines[-1][2] and lines[-1][1] == lines[-1][3]:
                            print(f"Zero-length line in line specification (file line {count}) -- skipping.")
                            lines.pop()
                    else:
                        print(f"Incorrect parameter count for line specifiers ({len(parts)}) in {fname}.")
    except FileNotFoundError:
        print(f"File '{fname}' not found.")
    except Exception as error:
        print(f"An error occurred while reading '{fname}': {error}")

    return angle, boxes, transcript, lines


def dist2seg(x, y, seg):
    """Distance from a point to the nearest point on a line segment."""
    x0, y0, x1, y1 = seg
    rx = x - x0
    ry = y - y0
    rx1 = x1 - x0
    ry1 = y1 - y0
    seglen = math.sqrt(rx1 ** 2 + ry1 ** 2)
    if seglen == 0:
        return math.sqrt(rx ** 2 + ry ** 2)

    parcomp = (rx * rx1 + ry * ry1) / seglen
    if parcomp < 0:
        return math.sqrt(rx ** 2 + ry ** 2)
    if parcomp > seglen:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    t = parcomp / seglen
    px = x0 + t * rx1
    py = y0 + t * ry1
    return math.sqrt((x - px) ** 2 + (y - py) ** 2)


def getRowBoxes(boxes, gangle=None, lines=None):
    """Order boxes into rows, following the contest evaluator's row logic."""
    lines = lines or []
    if len(lines) == 0:
        return infer_row_boxes(boxes, gangle=gangle)

    label = np.zeros(len(lines), dtype=np.int32)
    for i in range(1, len(lines)):
        if lines[i][0] == lines[i - 1][2] and lines[i][1] == lines[i - 1][3]:
            label[i] = label[i - 1]
        else:
            label[i] = label[i - 1] + 1

    blabel = np.zeros(len(boxes))
    for i, box in enumerate(boxes):
        d = dist2seg(*box.center(), lines[0])
        blabel[i] = 0
        for j in range(1, len(lines)):
            jd = dist2seg(*box.center(), lines[j])
            if jd < d:
                d = jd
                blabel[i] = label[j]

    rowlist = []
    idlist = []
    for row_label in range(label[-1] + 1):
        row = []
        rowid = []
        rank = []
        first_box = None
        for box_index, box_label in enumerate(blabel):
            if box_label != row_label:
                continue
            box = boxes[box_index]
            if first_box is None:
                first_box = box
            row.append(box)
            rowid.append(box_index)
            rank.append(
                (box.xnw - first_box.xnw) * (first_box.xne - first_box.xnw)
                + (box.ynw - first_box.ynw) * (first_box.yne - first_box.ynw)
            )
        if row:
            order = np.argsort(rank)
            rowlist.append([row[i] for i in order])
            idlist.append([rowid[i] for i in order])
    return rowlist, idlist


def infer_row_boxes(boxes, gangle=None):
    nboxes = len(boxes)
    bclust = np.full(nboxes, -1)
    bseq = np.zeros(nboxes)
    lnbr = np.full(nboxes, -1)
    rnbr = np.full(nboxes, -1)

    xcent = [box.center()[0] for box in boxes]
    xsort = np.sort(xcent)
    xord = np.argsort(xcent)
    xlut = np.argsort(xord)

    for i in range(nboxes):
        ci = boxes[i].center()
        angle = boxes[i].angle() if gangle is None else gangle
        ax = math.cos(angle)
        ay = math.sin(angle)
        rd = np.inf
        ld = np.inf
        klo = xlut[i]
        khi = klo
        while min(ci[0] - xsort[klo], xsort[khi] - ci[0]) < max(rd, ld) and (klo > 0 or khi < nboxes - 1):
            if (ci[0] - xsort[klo] < xsort[khi] - ci[0] or khi >= nboxes - 1) and klo > 0:
                klo -= 1
                j = xord[klo]
            elif khi < nboxes - 1:
                khi += 1
                j = xord[khi]
            else:
                break

            d = boxes[i].dist(boxes[j])
            cj = boxes[j].center()
            rj = boxes[j].radius()
            dotprod = (cj[0] - ci[0]) * ax + (cj[1] - ci[1]) * ay
            is_right = dotprod > 0
            if i != j and d < ld and not is_right:
                c = (ci[0] - d * ax, ci[1] - d * ay)
                if (cj[0] - c[0]) ** 2 + (cj[1] - c[1]) ** 2 < rj * rj:
                    ld = d
                    lnbr[i] = j
            if i != j and d < rd and is_right:
                c = (ci[0] + d * ax, ci[1] + d * ay)
                if (cj[0] - c[0]) ** 2 + (cj[1] - c[1]) ** 2 < rj * rj:
                    rd = d
                    rnbr[i] = j

    for i in range(len(rnbr)):
        if rnbr[i] >= 0 and lnbr[rnbr[i]] != i and lnbr[rnbr[i]] < 0:
            lnbr[rnbr[i]] = i
        if lnbr[i] >= 0 and rnbr[lnbr[i]] != i and rnbr[lnbr[i]] < 0:
            rnbr[lnbr[i]] = i

    clust = 0
    cid = []
    cseq = []
    for i in range(nboxes):
        angle = boxes[i].angle() if gangle is None else gangle
        if bclust[i] != -1:
            continue
        current = [i]
        bclust[i] = clust
        bseq[i] = boxes[i].center()[0] * math.cos(angle) + boxes[i].center()[1] * math.sin(angle)

        j = lnbr[i]
        while j >= 0 and j not in current:
            bclust[j] = clust
            current.append(j)
            bseq[j] = boxes[j].center()[0] * math.cos(angle) + boxes[j].center()[1] * math.sin(angle)
            j = lnbr[j]

        j = rnbr[i]
        while j >= 0 and j not in current:
            bclust[j] = clust
            current.append(j)
            bseq[j] = boxes[j].center()[0] * math.cos(angle) + boxes[j].center()[1] * math.sin(angle)
            j = rnbr[j]

        cseq.append(-boxes[i].center()[0] * math.sin(angle) + boxes[i].center()[1] * math.cos(angle))
        cid.append(current)
        clust += 1

    rowlist = []
    for current in cid:
        rowlist.append([boxes[current[i]] for i in np.argsort(bseq[current])])
    row_order = np.argsort(cseq)
    return [rowlist[i] for i in row_order], [cid[i] for i in row_order]


def orderBoxes(boxes, gangle=None, lines=None):
    rowlist, _ = getRowBoxes(boxes, gangle=gangle, lines=lines)
    return [box for row in rowlist for box in row]


def getRowTranscript(fname):
    """Return transcript rows in full-width left-to-right/top-down order."""
    gangle, boxes, transcript, lines = readBoxFile(fname)
    boxes2 = orderBoxes(boxes, gangle, lines)
    _, idlist = getRowBoxes(boxes2, gangle=gangle)
    lintran = "".join(transcript)
    return "".join("".join(lintran[i] for i in rowid) + "\n" for rowid in idlist)


def evaluate(prediction, gt):
    prediction = convert(prediction).upper().replace(" ", "").replace("\t", "")
    n_error = levenshtein(prediction, gt)
    n_char = len(gt)
    cer = n_error / n_char if n_char else 0.0
    return cer, n_error, n_char


def run_evaluations(pred_dir, gt_dir):
    gt_files = [
        name
        for name in listdir(gt_dir)
        if isfile(join(gt_dir, name)) and name.endswith("_Rotation1_300dpi_letters.txt")
    ]
    n_error = []
    n_char = []
    for gt_file in gt_files:
        pred_file = os.path.join(pred_dir, gt_file[: -len("_Rotation1_300dpi_letters.txt")] + "_transcript.txt")
        try:
            with open(pred_file, "r", encoding="utf-8") as handle:
                pred = "".join(line for line in handle)
        except FileNotFoundError:
            print(f"Unable to find prediction file '{pred_file}'.")
            pred = ""
        gt = getRowTranscript(os.path.join(gt_dir, gt_file))
        _, sq_n_error, sq_n_char = evaluate(pred, gt)
        n_error.append(sq_n_error)
        n_char.append(sq_n_char)
    return sum(n_error) / sum(n_char) if sum(n_char) else 0.0

