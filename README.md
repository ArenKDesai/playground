# Neural Network Playground: Predicting Electricity Prices

Live at **https://arenkdesai.github.io/playground/**

A modified version of TensorFlow's
[Neural Network Playground](https://github.com/tensorflow/playground). Instead
of classification toy datasets, the page opens with a neural network learning
to predict electricity prices (locational marginal prices, or LMPs) from the
hour of day. Press play and watch the prediction line bend into the daily
"duck curve": cheap overnight power, a morning ramp, negative prices at
midday when solar floods the grid, and an evening scarcity spike. The data is
simulated and deliberately exaggerated so the shape is easy to see.

The only other dataset is **Custom**, which opens an editor for your own data:

- **Formula:** type a target `f(x, y)` such as `sin(sqrt(x^2 + y^2))`, with
  x and y in [-6, 6]. The output is rescaled to [-1, 1]. Formulas are stored
  in the URL hash, so links are shareable. They are parsed by a small
  expression parser (no `eval`): arithmetic, `^`, the usual math functions,
  `pi` and `e`.
- **CSV data:** paste or upload rows of `x, y, target` (or `x, target` for
  1-D data). A header row is optional. Inputs are min-max scaled to the
  playground grid and the target to [-1, 1]. Files over 3000 rows are
  subsampled. CSV data is kept in the browser's local storage and is not
  uploaded or put in the URL. *Load example* fills in 24 hourly prices.

When the target only depends on x (the LMP example, a 2-column CSV, or a
formula without `y`), the output panel is a 1-D plot: each dot's height is its
target value, the black line is the network's prediction, the axes show the
original units, and the y features are switched off.

The LMP example and custom data live in `src/customdata.ts`. Everything else
is based on upstream [tensorflow/playground](https://github.com/tensorflow/playground).

Deep playground is an interactive visualization of neural networks, written in
TypeScript using d3.js. We use GitHub issues for tracking new requests and bugs.
Your feedback is highly appreciated!

**If you'd like to contribute, be sure to review the [contribution guidelines](CONTRIBUTING.md).**

## Development

To run the visualization locally, run:
- `npm i` to install dependencies
- `npm run build` to compile the app and place it in the `dist/` directory
- `npm run serve` to serve from the `dist/` directory and open a page on your browser.

For a fast edit-refresh cycle when developing run `npm run serve-watch`.
This will start an http server and automatically re-compile the TypeScript,
HTML and CSS files whenever they change.

## For owners
To push to production: `git subtree push --prefix dist origin gh-pages`.

This is not an official Google product.
