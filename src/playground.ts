/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  regDatasets,
  activations,
  regularizations,
  getKeyFromValue
} from "./state";
import {Example2D, shuffle} from "./dataset";
import * as customdata from "./customdata";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

let mainWidth;

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES = 800;
const DENSITY = 100;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === customdata.regressCustom) {
      openCustomEditor();
    }
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    updateCustomEditButton();
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);
  updateCustomEditButton();
  makeCustomEditor();

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    updateHeatmapPoints();
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property("value", state.regularizationRate);

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], false);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
          false);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 8) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);
  // For 1-D data the network never sees y, so evaluate it at y = 0.
  let oneD = isOneD();

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = oneD ? 0 : yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], false);
  if (isOneD()) {
    // Plot the prediction; every column of the output matrix is constant.
    heatMap.updateCurve(boundary[nn.getOutputNode(network).id]
        .map(column => column[0] * customdata.DISPLAY_RADIUS));
  } else {
    heatMap.updateCurve(null);
  }

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        false);
  });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup=false) {
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  network = nn.buildNetwork(shape, state.activation, nn.Activations.LINEAR,
      state.regularization, constructInputIds(), state.initZero);
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data = customdata.toDisplayPoints(data, isOneDGenerator(dataGenerator));
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      // Same orientation as the output heatmap (y up).
      context.fillRect(w * (d.x + 6) / 12, h * (6 - d.y) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  for (let regDataset in regDatasets) {
    let canvas: any =
        document.querySelector(`canvas[data-regDataset=${regDataset}]`);
    renderThumbnail(canvas, regDatasets[regDataset]);
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

/** Key used to keep uploaded/pasted CSV data across page loads. */
const CSV_STORAGE_KEY = "playground.customCsv";

function loadStoredCsv(): string {
  try {
    let text = window.localStorage.getItem(CSV_STORAGE_KEY);
    if (text != null && text.trim() !== "") {
      return text;
    }
  } catch (e) {
    // Local storage can be unavailable (e.g. privacy mode).
  }
  return customdata.exampleLmpCsv();
}

/** Returns false when the CSV could not be persisted. */
function storeCsv(text: string): boolean {
  try {
    window.localStorage.setItem(CSV_STORAGE_KEY, text);
    return true;
  } catch (e) {
    return false;
  }
}

/** Configures the custom regression dataset from the (URL) state. */
function initCustomData() {
  if (state.regSource === "csv") {
    try {
      customdata.setCsv(customdata.parseCsv(loadStoredCsv()));
      return;
    } catch (e) {
      // Fall back to the formula below.
    }
  }
  state.regSource = "formula";
  try {
    customdata.setFormula(customdata.prepareFormula(state.regFormula));
  } catch (e) {
    state.regFormula = customdata.DEFAULT_FORMULA;
    customdata.setFormula(customdata.prepareFormula(state.regFormula));
  }
}

/** Whether data from this generator only depends on x. */
function isOneDGenerator(generator: (n: number, noise: number) => Example2D[]):
    boolean {
  return generator === customdata.regressLmp ||
      (generator === customdata.regressCustom &&
      customdata.isOneDimensional());
}

/** Whether the current data only depends on x. */
function isOneD(): boolean {
  return isOneDGenerator(state.regDataset);
}

/** Features that depend on y, which carry no information for 1-D data. */
const Y_FEATURES = ["y", "ySquared", "xTimesY", "sinY"];
const X_FEATURES = ["x", "xSquared", "sinX"];
let lastOneD: boolean = null;

/**
 * Turns off the y features when switching to 1-D data, and makes sure one is
 * on when switching to 2-D data. Callers reset the network afterwards.
 */
function syncOneDFeatures() {
  let oneD = isOneD();
  if (oneD === lastOneD) {
    return;
  }
  lastOneD = oneD;
  let has = (ids: string[]) => ids.some(id => id in INPUTS && state[id]);
  if (oneD) {
    Y_FEATURES.forEach(id => state[id] = false);
    if (!has(X_FEATURES) && "x" in INPUTS) {
      state.x = true;
    }
  } else if (!has(Y_FEATURES) && "y" in INPUTS) {
    state.y = true;
  }
}

function updateCustomEditButton() {
  let show = state.regDataset === customdata.regressCustom;
  d3.select("#custom-edit-button").style("display", show ? "block" : null);
}

let openCustomEditor: () => void;

/** Wires up the modal for editing the custom regression dataset. */
function makeCustomEditor() {
  let modal = d3.select("#custom-data-modal");
  let formulaInput = d3.select("#custom-formula");
  let csvInput = d3.select("#custom-csv");
  let status = d3.select("#custom-status");
  let applyButton = d3.select("#custom-apply");
  let canvas = document.getElementById("custom-preview-canvas") as
      HTMLCanvasElement;

  let draftSource: customdata.Source = "formula";
  let draftFormula: customdata.FormulaInfo = null;
  let draftCsv: customdata.CsvData = null;

  function fmt(v: number): string {
    return String(+v.toPrecision(4));
  }

  function setStatus(text: string, isError: boolean) {
    status.text(text).classed("error", isError);
    applyButton.property("disabled", isError);
  }

  function drawPreview() {
    let ctx = canvas.getContext("2d");
    let w = canvas.width, h = canvas.height;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, w, h);
    if (draftSource === "formula" && draftFormula != null) {
      // Fill the whole square with the (rescaled) function value.
      let cell = 4;
      let [lo, hi] = draftFormula.range;
      for (let i = 0; i < w; i += cell) {
        for (let j = 0; j < h; j += cell) {
          let x = (i + cell / 2) / w * 12 - 6;
          let y = 6 - (j + cell / 2) / h * 12;
          let v = draftFormula.fn(x, y);
          if (!isFinite(v)) {
            continue;
          }
          let label = hi - lo < 1e-12 ? 0 : (v - lo) / (hi - lo) * 2 - 1;
          ctx.fillStyle = colorScale(label) as any;
          ctx.fillRect(i, j, cell, cell);
        }
      }
      if (draftFormula.oneDimensional) {
        // Plot f(x) on top of the stripes.
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i <= w; i++) {
          let v = draftFormula.fn(i / w * 12 - 6, 0);
          let label = hi - lo < 1e-12 ? 0 : (v - lo) / (hi - lo) * 2 - 1;
          let py = h * (6 - label * customdata.DISPLAY_RADIUS) / 12;
          if (i === 0) {
            ctx.moveTo(i, py);
          } else {
            ctx.lineTo(i, py);
          }
        }
        ctx.stroke();
      }
    } else if (draftSource === "csv" && draftCsv != null) {
      let points = customdata.toDisplayPoints(
          customdata.generate("csv", null, draftCsv, 0, 0),
          draftCsv.oneDimensional);
      points.forEach(d => {
        ctx.fillStyle = colorScale(d.label) as any;
        ctx.fillRect(w * (d.x + 6) / 12 - 2, h * (6 - d.y) / 12 - 2, 4, 4);
      });
    }
  }

  function validate() {
    draftFormula = null;
    draftCsv = null;
    try {
      if (draftSource === "formula") {
        draftFormula =
            customdata.prepareFormula(formulaInput.property("value"));
        let [lo, hi] = draftFormula.range;
        setStatus(`Range on the grid: [${fmt(lo)}, ${fmt(hi)}]` +
            (hi - lo < 1e-12 ? " (constant)" : ""), false);
      } else {
        let csv = customdata.parseCsv(csvInput.property("value"));
        draftCsv = csv;
        let c = csv.columns;
        let parts = [`${csv.numRows} rows`];
        if (csv.numSkipped > 0) {
          parts.push(`${csv.numSkipped} skipped`);
        }
        parts.push(`x = ${c[0]} [${fmt(csv.xRange[0])}, ` +
            `${fmt(csv.xRange[1])}]`);
        if (!csv.oneDimensional) {
          parts.push(`y = ${c[1]} [${fmt(csv.yRange[0])}, ` +
              `${fmt(csv.yRange[1])}]`);
        }
        parts.push(`target = ${c[c.length - 1]} ` +
            `[${fmt(csv.targetRange[0])}, ${fmt(csv.targetRange[1])}]`);
        if (csv.numRows > customdata.MAX_CSV_POINTS) {
          parts.push(`subsampled to ${customdata.MAX_CSV_POINTS}`);
        }
        setStatus(parts.join(" · "), false);
      }
    } catch (e) {
      setStatus(e.message, true);
    }
    drawPreview();
  }

  function selectSource(source: customdata.Source) {
    draftSource = source;
    d3.selectAll(".custom-tab").classed("active", function() {
      return this.getAttribute("data-source") === source;
    });
    d3.selectAll(".custom-pane").classed("active", function() {
      return this.getAttribute("data-pane") === source;
    });
    validate();
  }

  function close() {
    modal.classed("open", false);
  }

  function apply() {
    let note = "";
    if (draftSource === "formula") {
      if (draftFormula == null) {
        return;
      }
      state.regFormula = formulaInput.property("value").trim();
      customdata.setFormula(draftFormula);
    } else {
      if (draftCsv == null) {
        return;
      }
      customdata.setCsv(draftCsv);
      if (!storeCsv(csvInput.property("value"))) {
        note = "This CSV is too large to save in the browser, so it will " +
            "be lost when the page reloads.";
      }
    }
    state.regSource = draftSource;
    state.regDataset = customdata.regressCustom;
    d3.selectAll("canvas[data-regDataset]").classed("selected", false);
    d3.select("canvas[data-regDataset=custom]").classed("selected", true);
    close();
    if (note) {
      alert(note);
    }
    drawDatasetThumbnails();
    updateCustomEditButton();
    generateData();
    parametersChanged = true;
    reset();
  }

  openCustomEditor = () => {
    formulaInput.property("value", state.regFormula);
    csvInput.property("value", loadStoredCsv());
    modal.classed("open", true);
    selectSource(customdata.getSource());
    if (draftSource === "formula") {
      (formulaInput.node() as HTMLInputElement).focus();
    }
  };

  d3.select("#custom-edit-button").on("click", () => openCustomEditor());
  d3.selectAll(".custom-tab").on("click", function() {
    selectSource(this.getAttribute("data-source"));
  });
  formulaInput.on("input", validate);
  formulaInput.on("keydown", () => {
    if ((d3.event as KeyboardEvent).keyCode === 13 &&
        !applyButton.property("disabled")) {
      apply();
    }
  });
  csvInput.on("input", validate);
  d3.selectAll(".formula-presets button").on("click", function() {
    formulaInput.property("value", this.getAttribute("data-formula"));
    validate();
  });
  d3.select("#custom-csv-example").on("click", () => {
    csvInput.property("value", customdata.exampleLmpCsv());
    validate();
  });
  d3.select("#custom-csv-clear").on("click", () => {
    csvInput.property("value", "");
    validate();
  });
  let fileInput = d3.select("#custom-csv-file");
  d3.select("#custom-csv-upload").on("click", () => {
    (fileInput.node() as HTMLInputElement).click();
  });
  fileInput.on("change", function() {
    let file: File = this.files && this.files[0];
    if (file == null) {
      return;
    }
    let reader = new FileReader();
    reader.onload = () => {
      csvInput.property("value", reader.result as string);
      validate();
    };
    reader.onerror = () => setStatus(`Could not read ${file.name}`, true);
    reader.readAsText(file);
    // Allow re-uploading the same file.
    this.value = "";
  });
  applyButton.on("click", apply);
  d3.select("#custom-cancel").on("click", close);
  modal.on("click", function() {
    // Close when clicking the backdrop, not the card.
    if ((d3.event as Event).target === this) {
      close();
    }
  });
  d3.select(document).on("keydown.custom", () => {
    if ((d3.event as KeyboardEvent).keyCode === 27 &&
        modal.classed("open")) {
      close();
    }
  });
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let data = state.regDataset(NUM_SAMPLES, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  syncOneDFeatures();
  updateHeatmapPoints();
}

function updateHeatmapPoints() {
  let oneD = isOneD();
  heatMap.updatePoints(customdata.toDisplayPoints(trainData, oneD));
  heatMap.updateTestPoints(state.showTestData ?
      customdata.toDisplayPoints(testData, oneD) : []);
  let axes = state.regDataset === customdata.regressLmp ?
      customdata.LMP_AXES : customdata.axisDomains();
  heatMap.setAxisDomains(axes.x, axes.y, axes.xLabel, axes.yLabel);
  d3.select("#oned-note").style("display", oneD ? null : "none");
  if (!oneD) {
    heatMap.updateCurve(null);
  }
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', {'sessionControl': 'start'});
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

initCustomData();
drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
