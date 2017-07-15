import GraphUtil from './graph-util';
import {Rapahel} from 'react-raphael';

const Util = {

    rawData: function (options) {
        const gridRange = options.gridRange || options.range;
        const gridStep = options.gridStep || [1, 1];
        const range = options.range || [[-10, 10], [-10, 10]];
        const scale = options.scale || [20, 20]
        const ticks = options.ticks != null ? options.ticks : true;
        const tickStep = options.tickStep || [2, 2];
        const axisCenter = options.axisCenter || [
                Math.min(Math.max(range[0][0], 0), range[0][1]),
                Math.min(Math.max(range[1][0], 0), range[1][1]),
            ];
        const halfWidthTicks = options.halfWidthTicks || false;
        const labelStep = options.labelStep || [1, 1];
        const unityLabels = options.unityLabels || false;
        const axisArrows = options.axisArrows || "";
        const tickLen = options.tickLen || [5, 5];
        const gridOpacity = options.gridOpacity || 0.1;
        const axisOpacity = options.axisOpacity || 1.0;
        const tickOpacity = options.tickOpacity || 1.0;
        const labelOpacity = options.labelOpacity || 1.0;
        return {
            range: range,
            gridRange: gridRange,
            gridStep: gridStep,
            scale: scale,
            ticks: ticks,
            tickStep: tickStep,
            axisCenter: axisCenter,
            halfWidthTicks: halfWidthTicks,
            labelStep: labelStep,
            unityLabels: unityLabels,
            axisArrows: axisArrows,
            tickLen: tickLen,
            gridOpacity: gridOpacity,
            axisOpacity: axisOpacity,
            tickOpacity: tickOpacity,
            labelOpacity: labelOpacity,



        }
    },

    //画布数据
    canvasData: function (options) {
        let scale = options.scale || [40, 40];
        scale = (typeof scale === "number" ? [scale, scale] : scale);

        const xScale = scale[0];
        const yScale = scale[1];

        if (options.range == null) {
            return Khan.error("range should be specified in graph init");
        }

        const xRange = options.range[0];
        const yRange = options.range[1];

        const width = (xRange[1] - xRange[0]) * xScale;
        const height = (yRange[1] - yRange[0]) * yScale;
        return {
            width: width,
            height: height
        }

    },

    //网格数据
    gridData: function (options) {
        const {gridRange, gridStep, range, scale, gridOpacity} = Util.rawData(options);

        const xr = gridRange[0];
        const yr = gridRange[1];
        let xPoints = [];
        let yPoints = [];
        let x = gridStep[0] * Math.ceil(xr[0] / gridStep[0]);
        for (; x <= xr[1]; x += gridStep[0]) {
            let start = GraphUtil.scalePoint([x, yr[0]],range, scale);
            let end = GraphUtil.scalePoint([x, yr[1]],range, scale);
            xPoints.push([start, end]);
        }
        let y = gridStep[1] * Math.ceil(yr[0] / gridStep[1]);
        for (; y <= yr[1]; y += gridStep[1]) {
            let start = GraphUtil.scalePoint([xr[0], y],range, scale);
            let end = GraphUtil.scalePoint([xr[1], y],range, scale);
            yPoints.push([start, end]);
        }

        const attr = {
            opacity: gridOpacity,
            stroke:  "#000000",
            strokeWidth: 2
        };

        return {
            xGridPoints: xPoints,
            yGridPoints: yPoints,
            attr: attr
        }
    },

    //绘制坐标的数据
    axisData: function (options) {

        const {gridRange, range, scale, axisCenter, axisOpacity} = Util.rawData(options);
        let xstart = GraphUtil.scalePoint([gridRange[0][0], axisCenter[1]], range, scale),
            xend = GraphUtil.scalePoint([gridRange[0][1], axisCenter[1]], range, scale),
            ystart = GraphUtil.scalePoint([axisCenter[0], gridRange[1][0]], range, scale),
            yend = GraphUtil.scalePoint([axisCenter[0], gridRange[1][1]], range, scale);

        const attr = {
            opacity: axisOpacity,
            stroke:  "#000000",
            strokeWidth: 2
        };

        return {
            xAxisPoint: [xstart, xend],
            yAxisPoint: [ystart, yend],
            attr: attr
        }

    },

    tickData: function (options) {
        const {gridRange, gridStep, range, scale, tickStep, tickLen, axisCenter, halfWidthTicks, axisArrows, tickOpacity} = Util.rawData(options);
        let step = gridStep[0] * tickStep[0];
        let len = tickLen[0] / scale[1];
        let start = gridRange[0][0];
        let stop = gridRange[0][1];

        let xPlusPoints = [];
        let xMinusPoints = [];
        let yPlusPoinsts = [];
        let yMinusPoints = [];
        if (range[1][0] < 0 && range[1][1] > 0) {
            for (let x = step + axisCenter[0]; x <= stop; x += step) {
                if (x < stop || !axisArrows) {
                    let xstart = GraphUtil.scalePoint([x, -len + axisCenter[1]], range, scale);
                    let xend = GraphUtil.scalePoint([x, halfWidthTicks ? 0 : len + axisCenter[1]], range, scale)
                    xPlusPoints.push([xstart, xend])
                }

            }
            for (let x = -step + axisCenter[0]; x >= start; x -= step) {
                if (x > start || !axisArrows) {
                    let xstart01 = GraphUtil.scalePoint([x, -len + axisCenter[1]], range, scale);
                    let xend01 = GraphUtil.scalePoint([x, halfWidthTicks ? 0 : len + axisCenter[1]], range, scale);
                    xMinusPoints.push([xstart01, xend01])
                }
            }
        }

            // vertical axis
            step = gridStep[1] * tickStep[1];
            len = tickLen[1] / scale[0];
            start = gridRange[1][0];
            stop = gridRange[1][1];

            if (range[0][0] < 0 && range[0][1] > 0) {
                for (let y = step + axisCenter[1]; y <= stop; y += step) {
                    if (y < stop || !axisArrows) {

                        let ystart = GraphUtil.scalePoint([-len + axisCenter[0], y], range, scale);
                        let yend = GraphUtil.scalePoint([halfWidthTicks ? 0 : len + axisCenter[0], y], range, scale)
                        yPlusPoinsts.push([ystart, yend])
                    }
                }

                for (let y = -step + axisCenter[1]; y >= start; y -= step) {
                    if (y > start || !axisArrows) {
                        let ystart = GraphUtil.scalePoint([-len + axisCenter[0], y], range, scale);
                        let yend = GraphUtil.scalePoint([halfWidthTicks ? 0 : len + axisCenter[0], y], range, scale);
                        yMinusPoints.push([ystart, yend]);
                    }
                }
            }

        const attr = {
            opacity: tickOpacity,
            stroke:  "#000000",
            strokeWidth: 1
        };

        return {
            xTickPoints: [xPlusPoints, xMinusPoints],
            yTickPoints: [yPlusPoinsts, yMinusPoints],
            attr: attr
        }
    },

    labelData: function (options) {
        const {gridRange, gridStep, range, scale, tickStep, labelStep, axisCenter, unityLabels, axisArrows, labelOpacity} = Util.rawData(options);
        let step = gridStep[0] * tickStep[0] * labelStep[0];
        let start = gridRange[0][0];
        let stop = gridRange[0][1];
        const xAxisPosition = (axisCenter[0] < 0) ? "above" : "below";
        const yAxisPosition = (axisCenter[1] < 0) ? "right" : "left";
        const xShowZero = axisCenter[0] === 0 && axisCenter[1] !== 0;
        const yShowZero = axisCenter[0] !== 0 && axisCenter[1] === 0;
        const axisOffCenter = axisCenter[0] !== 0 ||
            axisCenter[1] !== 0;
        const showUnityX = unityLabels[0] || axisOffCenter;
        const showUnityY = unityLabels[1] || axisOffCenter;

        let xPoints = [];
        let yPoints = [];

        // positive x-axis
        for (let x = (xShowZero ? 0 : step) + axisCenter[0]; x <= stop; x += step) {
            if (x < stop || !axisArrows) {
                let xpoint = GraphUtil.scalePoint([x, axisCenter[1]], range, scale);
                let text = "\\small{" + x + "}";
                xPoints.push({point: xpoint, text: text, direction: xAxisPosition});
            }
        }

        // negative x-axis
        for (let x = -step + axisCenter[0]; x >= start; x -= step) {
            if (x > start || !axisArrows) {
                let xpoint = GraphUtil.scalePoint([x, axisCenter[1]], range, scale);
                let text= "\\small{" + x + "}";
                xPoints.push({point: xpoint, text: text, direction: xAxisPosition});
            }
        }

        step = gridStep[1] * tickStep[1] * labelStep[1];
        start = gridRange[1][0];
        stop = gridRange[1][1];

        // positive y-axis
        for (let y = (yShowZero ? 0 : step) + axisCenter[1]; y <= stop; y += step) {
            if (y < stop || !axisArrows) {
                let ypoint = GraphUtil.scalePoint([axisCenter[0], y], range, scale);
                let text = "\\small{" + y + "}";
                yPoints.push({point: ypoint, text: text, direction: yAxisPosition});
            }
        }

        // negative y-axis
        for (let y = -step  + axisCenter[1]; y >= start; y -= step) {
            let ypoint = GraphUtil.scalePoint([axisCenter[0], y], range, scale);
            let text = "\\small{" + y + "}";
            yPoints.push({point: ypoint, text: text, direction: yAxisPosition});
            }

        const attr = {
            opacity: labelOpacity,
            stroke:  "#000000",
        };

        return {
            xLabelPoint: xPoints,
            yLabelPoint: yPoints,
            attr: attr
        }
    },

};

export default Util
