/**
 * Created by wangwei on 17/7/11.
 */

import {KhanMath}  from "../../../../util/math";

let GraphUtil = {

    scaleVector: function(point, scale) {
        if (typeof point === "number") {
            return GraphUtil.scaleVector([point, point]);
        }
        const xScale = scale[0];
        const yScale = scale[1];
        const x = point[0];
        const y = point[1];
        return [x * xScale, y * yScale];
    },

    scalePoint : function(point, range, scale) {
        if (typeof point === "number") {
            return GraphUtil.scalePoint([point, point]);
        }

        const x = point[0];
        const y = point[1];
        const xRange = range[0];
        const yRange = range[0];
        const xScale = scale[0];
        const yScale = scale[1];
        return [(x - xRange[0]) * xScale, (yRange[1] - y) * yScale];
    },

    unscalePoint: function(point, scale, range) {
        if (typeof point === "number") {
            return GraphUtil.unscalePoint([point, point], scale, range);
        }
        const xScale = scale[0];
        const yScale = scale[1];
        const x = point[0];
        const y = point[1];
        const xRange = range[0];
        const yRange = range[1];
        return [x / xScale + xRange[0], yRange[1] - y / yScale];
    },

    unscaleVector: function(point, scale) {
        if (typeof point === "number") {
            return GraphUtil.unscaleVector([point, point], scale);
        }
        const xScale = scale[0];
        const yScale = scale[1];
        return [point[0] / xScale, point[1] / yScale];
    },

    svgPath: function(points, alreadyScaled) {

        return points.map(function(point, i) {
            if (point === true) {
                return "z";
            } else {
                return (i === 0 ? "M" : "L") +
                    KhanMath.bound(point[0]) + " " + KhanMath.bound(point[1]);
            }
        }).join("");
    },

    /**
     * For a graph's x or y dimension, given the tick step,
     * the ranges extent (e.g. [-10, 10]), the pixel dimension constraint,
     * and the grid step, return a bunch of configurations for that dimension.
     *
     * Example:
     *      gridDimensionConfig(10, [-50, 50], 400, 5)
     *
     * Returns: {
     *      scale: 4,
     *      snap: 2.5,
     *      tickStep: 2,
     *      unityLabel: true
     * };
     */

    gridDimensionConfig: function(absTickStep, extent, dimensionConstraint,
                                  gridStep) {
        let scale = GraphUtil.scaleFromExtent(extent, dimensionConstraint);
        let stepPx = absTickStep * scale;
        let unityLabel = stepPx > 30;
        return {
            scale: scale,
            tickStep: absTickStep / gridStep,
            unityLabel: unityLabel
        }
    },

    /**
     * Given the range, step, and boxSize, calculate the reasonable gridStep.
     * Used for when one was not given explicitly.
     *
     * Example:
     *      getGridStep([[-10, 10], [-10, 10]], [1, 1], 340)
     *
     * Returns: [1, 1]
     */
    getGridStep: function(range, step, boxSize) {
        return _(2).times(function(i) {
            let scale = GraphUtil.scaleFromExtent(range[i], boxSize);
            return GraphUtil.gridStepFromTickStep(step[i], scale);

        });
    },

    /**
     * Given the range and a dimension, come up with the appropriate
     * scale.
     * Example:
     *      scaleFromExtent([-25, 25], 500) // returns 10
     */
    scaleFromExtent: function(extent, dimensionConstraint) {
        let span = extent[1] - extent[0];
        return dimensionConstraint / span;
    },

    /**
     * Return a reasonable tick step given extent and dimension.
     * (extent is [begin, end] of the domain.)
     * Example:
     *      tickStepFromExtent([-10, 10], 300) // returns 2
     */
    tickStepFromExtent: function(extent, dimensionConstraint) {
        let span = extent[1] - extent[0];

        let tickFactor;
        // If single number digits
        if (15 < span && span <= 20) {
            tickFactor = 23;

            // triple digit or decimal
        } else if (span > 100 || span < 5) {
            tickFactor = 10;

            // double digit
        } else {
            tickFactor = 16;
        }
        let constraintFactor = dimensionConstraint / 500;
        let desiredNumTicks = tickFactor * constraintFactor;
        return GraphUtil.tickStepFromNumTicks(span, desiredNumTicks);
    },

    /**
     * Given the tickStep and the graph's scale, find a
     * grid step.
     * Example:
     *      gridStepFromTickStep(200, 0.2) // returns 100
     */
    gridStepFromTickStep: function(tickStep, scale) {
        let tickWidth = tickStep * scale;
        let x = tickStep;
        let y = Math.pow(10, Math.floor(Math.log(x) / Math.LN10));
        let leadingDigit = Math.floor(x / y);
        if (tickWidth < 25) {
            return tickStep;
        }
        if (tickWidth < 50) {
            if (leadingDigit === 5) {
                return tickStep;
            } else {
                return tickStep / 2;
            }
        }
        if (leadingDigit === 1) {
            return tickStep / 2;
        }
        if (leadingDigit === 2) {
            return tickStep / 4;
        }
        if (leadingDigit === 5) {
            return tickStep / 5;
        }
    },

    /**
     * Find a good tick step for the desired number of ticks in the range
     * Modified from d3.scale.linear: d3_scale_linearTickRange.
     * Thanks, mbostock!
     * Example:
     *      tickStepFromNumTicks(50, 6) // returns 10
     */
    tickStepFromNumTicks: function(span, numTicks) {
        let step = Math.pow(10, Math.floor(Math.log(span / numTicks) / Math.LN10));
        let err = numTicks / span * step;

        // Filter ticks to get closer to the desired count.
        if (err <= 0.15) {
            step *= 10;
        } else if (err <= 0.35) {
            step *= 5;
        } else if (err <= 0.75) {
            step *= 2;
        }
        // Round start and stop values to step interval.
        return step;
    },

    getArrowHeads(path){

    }

};

export default GraphUtil