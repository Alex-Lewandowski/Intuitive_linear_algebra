# Intuitive Linear Algebra

An intuition-first linear algebra game. Every problem is generated fresh and small enough
to do in your head.

Play: https://alex-lewandowski.github.io/intuitive_linear_algebra/

## Chapters

1. **Matrix Machines**: a 2×2 matrix is a machine that moves the plane.
   Land it · Find it · Say it · Undo it

## Layout

```
index.html                       home page: chapter list + progress
shared/style.css                 shared look
shared/engine.js                 2×2 math, randomness, SVG drawing, dragging, progress
chapters/01-matrix-machines/     chapter 1 page + game.js (problem generators)
```

No build step: GitHub Pages serves the files as they are. Progress is saved in the
browser's localStorage under `ila-progress`.
