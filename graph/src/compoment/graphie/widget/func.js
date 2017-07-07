/**
 * Created by wangwei on 17/7/4.
 */
import {GraphieClasses} from './graphie-classes';
import MovablePoint from './movable-point';


var Parabola = GraphieClasses.createClass({
    displayName: "Parabola",

    movableProps: ["children"],

    getCoefficients: function(coords) {
        var p1 = coords[0];
        var p2 = coords[1];

        // Parabola with vertex (h, k) has form: y = a * (h - k)^2 + k
        var h = p1[0];
        var k = p1[1];

        // Use these to calculate familiar a, b, c
        var a = (p2[1] - k) / ((p2[0] - h) * (p2[0] - h));
        var b = - 2 * h * a;
        var c = a * h * h + k;

        return [a, b, c];
    },

    getFunctionForCoeffs: function(coeffs, x) {
        var a = coeffs[0], b = coeffs[1], c = coeffs[2];
        return (a * x + b) * x + c;
    },

    add: function(graphie) {
        let props = this.props;
        let coeffs = this.getCoefficients(this.props.coords);
        this.graphie = graphie;
        this.parabola = this.graphie.parabola(coeffs[0],coeffs[1], coeffs[1],
            props.style);
    },

    modify: function() {
        let props = this.props;
        let coeffs = this.getCoefficients(this.props.coords);
        let path = this.graphie.svgParabolaPath(coeffs[0], coeffs[1], coeffs[2]);
        this.parabola.attr(_.extend({}, props.style, { path: path }));
    },

    remove: function() {
        this.parabola.remove();
    },

    toFront: function() {
        this.parabola.toFront();
    },

    draw(){


    },

    addMovablePoint(graphie, coords){
        return new MovablePoint(graphie, coords)


    },
});

export  default Parabola