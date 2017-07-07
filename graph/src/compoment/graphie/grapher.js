/**
 * Created by wangwei on 17/7/4.
 */
import React, { Component } from 'react';
import GrapherWrapper from './grapher-wrapper';
import Parabola from './widget/func';
import _ from 'underscore'
import Util from '../../util/common';


class Grapher extends Component{

    constructor(props){
        super(props);
        this.state = {}
    }

    _setupGrapher(graphie, options){
        let gridConfigs = this._getGridConfig(options);
        if (options.markings === "graph") {
            graphie.graphInit({
                range: options.range,
                scale: _.pluck(gridConfigs, "scale"),
                axisArrows: "<->",
                labelFormat: function(s) { return "\\small{" + s + "}"; },
                gridStep: options.gridStep,
                snapStep: options.snapStep,
                tickStep: _.pluck(gridConfigs, "tickStep"),
                labelStep: 1,
                unityLabels: _.pluck(gridConfigs, "unityLabel"),
            });
            graphie.label([0, options.range[1][1]], options.labels[1], "above");
            graphie.label([options.range[0][1], 0], options.labels[0], "right");
        } else if (options.markings === "grid") {
            graphie.graphInit({
                range: options.range,
                scale: _.pluck(gridConfigs, "scale"),
                gridStep: options.gridStep,
                axes: false,
                ticks: false,
                labels: false,
            });
        } else if (options.markings === "none") {
            graphie.init({
                range: options.range,
                scale: _.pluck(gridConfigs, "scale")
            });
        }
    }

    _getGridConfig(options) {
        return _.map(options.step, function(step, i) {
            return Util.gridDimensionConfig(
                step,
                options.range[i],
                options.box[i],
                options.gridStep[i]);
        });
    }


    render(){
        let pointForCoord = (coord, i) => {
            return <MovablePoint
                key={i}
                coord={coord}
                static={this.props.static}
                constraints={[
                    Interactive2.MovablePoint.constraints.bound(),
                    Interactive2.MovablePoint.constraints.snap(),
                    (coord) => {
                        // Always enforce that this is a function
                        var isFunction = _.all(this._coords(),
                            (otherCoord, j) => {
                                return i === j  || !otherCoord ||
                                    !knumber.equal(coord[0], otherCoord[0]);
                            });
                        // Evaluate this criteria before per-point constraints
                        if (!isFunction) {
                            return false;
                        }
                        // Specific functions have extra per-point constraints
                        if (this.props.model &&
                            this.props.model.extraCoordConstraint) {
                            var extraConstraint =
                                this.props.model.extraCoordConstraint;
                            // Calculat resulting coords and verify that
                            // they're valid for this graph
                            var proposedCoords = _.clone(this._coords());
                            var oldCoord = _.clone(proposedCoords[i]);
                            proposedCoords[i] = coord;
                            return extraConstraint(coord, oldCoord,
                                proposedCoords, this._asymptote(),
                                this.props.graph);
                        }
                        return isFunction;
                    }
                ]}
                onMove={(newCoord, oldCoord) => {
                    var coords;
                    // Reflect over asymptote, if allowed
                    var asymptote = this._asymptote();
                    if (asymptote &&
                        this.props.model.allowReflectOverAsymptote &&
                        isFlipped(newCoord, oldCoord, asymptote)) {
                        coords = _.map(this._coords(), (coord) => {
                            return kpoint.reflectOverLine(coord, asymptote);
                        });
                    } else {
                        coords = _.clone(this._coords());
                    }
                    coords[i] = newCoord;
                    this.props.onChange({
                        coords: coords
                    });
                }}
                showHairlines={this.props.showHairlines}
                hideHairlines={this.props.hideHairlines}
                showTooltips={this.props.showTooltips}
                isMobile={this.props.isMobile}
            />;
        };

        let points = _.map(this._coords(), pointForCoord);

        return(
            <GrapherWrapper
                box={this.props.graph.box}
                range={this.props.graph.range}
                options={this.props.graph}
                setup={(graphie, options)=>this._setupGrapher(graphie, options)}
            >
                <Parabola
                    coords={[[0,0], [0.5, 0.5], [2,1]]}
                />
                {points}

            </GrapherWrapper>
        )
    }
}

export default Grapher;