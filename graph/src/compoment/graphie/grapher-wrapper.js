/**
 * Created by wangwei on 17/6/29.
 */
import React, { Component } from 'react';
import Modal from 'react-bootstrap';
import GraphUtils from '../../util/graphie';
// import InteractiveUtils from '../../util/interactive';
import Util from '../../util/common';
import _ from 'underscore';

let nestedMap = Util.nestedMap;
let deepEq = Util.deepEq;
let createGraphie = GraphUtils.createGraphie;

/* Widget and editor. */
class GrapherWrapper extends Component {

    constructor(props) {
        super(props);
        this.state = {}
    }

    componentDidMount() {
        this._setupGraphie();
        this._updateMovables();
    }

    shouldComponentUpdate(nextProps) {
        return !deepEq(this.props, nextProps);
    }

    componentDidUpdate(prevProps) {
        // If someone changes the setup function passed in, we should
        // technically setup graphie again. But that's definitely an
        // anti-pattern, since it is most-likely caused by passing in an
        // anonymous function rather than a "real" change, and re-rendering
        // in that case would cause us to constantly re-setup graphie, which
        // would have horrible performance implications. In order to avoid
        // those, we just warn here.
        if (this.props.setup !== prevProps.setup &&
            window.console && window.console.warn) {
            window.console.warn("<Graphie> was given a new setup function. " +
                "This is a bad idea; please refactor your code to give " +
                "the same setup function reference to <Graphie> on " +
                "every render.");
        }
        if (!deepEq(this.props.options, prevProps.options) ||
            !deepEq(this.props.box, prevProps.box) ||
            !deepEq(this.props.range, prevProps.range)) {
            this._setupGraphie();
        }
        this._updateMovables();
    }

    _renderMovables(children, options) {
    // Each leaf of `children` is a movable descriptor created by a call to
    // some `GraphieMovable`, such as `MovablePoint`.
    //
    // This function takes these descriptors and renders them into
    // on-screen movables, or updates on-screen movables for
    // descriptors when possible.
    //
    // If there is no movable with that key already, this descriptor is
    // stored in this._movables and promoted to an on-screen movable by
    // calling `child.add(graphie)`.
    //
    // If a movable of the same type with the same key exists already,
    // we take `child.props` and give them to the already-existing
    // on-screen movable, and call `movable.modify()`
    let graphie = options.graphie;
    let oldMovables = options.oldMovables;
    let newMovables = options.newMovables; /* output parameter */

    let renderChildren = (elem) => {
        _.each(elem.movableProps, (prop) => {
            // Render the children, and save the results of that
            // render to the appropriate props
            elem.props[prop] = this._renderMovables(
                elem.props[prop],
                options
            );
        });
    };

    // Add/modify movables

    // We want to keep track of whether we have added a new svg element,
    // because if we have, then we need to call .toFront() on any svg
    // elements occurring afterwards. If this happens, we set
    // `areMovablesOutOfOrder` to true:
    let areMovablesOutOfOrder = false;
    return nestedMap(children, (childDescriptor) => {
        if (!childDescriptor) {
            // Still increment the key to avoid cascading key changes
            // on hiding/unhiding children, i.e. by using
            // {someBoolean && <MovablePoint />}
            options.nextKey++;
            // preserve the null/undefined in the resulting array
            return childDescriptor;
        }

        // Instantiate the descriptor to turn it into a real Movable
        var child = new childDescriptor.type(childDescriptor.props);
        // assert(child instanceof GraphieMovable,
        //     "All children of a Graphie component must be Graphie " +
        //     "movables");

        // Give each child a key
        var keyProp = childDescriptor.key;
        var key = (keyProp == null) ?
            ("_no_id_" + options.nextKey) :
            keyProp;
        options.nextKey++;
        var ref = childDescriptor.ref;

        // We render our children first. This allows us to replace any
        // `movableProps` on our child with the on-screen movables
        // corresponding with those descriptors.
        renderChildren(child);

        var prevMovable = oldMovables[key];
        if (!prevMovable) {
            // We're creating a new child
            child.add(graphie);
            areMovablesOutOfOrder = true;

            newMovables[key] = child;

        } else if (child.constructor === prevMovable.constructor) {
            // We're updating an old child
            prevMovable.props = child.props;
            var modifyResult = prevMovable.modify(graphie);
            if (modifyResult === "reordered") {
                areMovablesOutOfOrder = true;
            }

            newMovables[key] = prevMovable;

        } else {
            // We're destroying an old child and replacing it
            // with a new child of a different type

            // This generally is a bad idea, so warn about it if this
            // is being caused by implicit keys
            if (keyProp == null) {
                if (typeof console !== "undefined" &&
                    console.warn) { // @Nolint
                    console.warn("Replacing a <Graphie> child with a " + // @Nolint
                        "child of a different type. Please add keys " +
                        "to your <Graphie> children");
                }
            }

            prevMovable.remove();
            child.add(graphie);
            areMovablesOutOfOrder = true;

            newMovables[key] = child;
        }

        if (areMovablesOutOfOrder) {
            newMovables[key].toFront();
        }

        if (ref) {
            this.movables[ref] = newMovables[key];
        }

        return newMovables[key];
    });
}

    // Sort of like react diffing, but for movables
    _updateMovables() {
        let graphie = this._graphie;

        let oldMovables = this._movables;
        let newMovables = {};
        this._movables = newMovables;
        this.movables = {};

        this._renderMovables(this.props.children, {
            nextKey: 1,
            graphie: graphie,
            oldMovables: oldMovables,
            newMovables: newMovables,
        });

        // Remove any movables that no longer exist in the child array
        _.each(oldMovables, (oldMovable, key) => {
            if (!newMovables[key]) {
                oldMovable.remove();
            }
        });
    }

    _setupGraphie() {
        this._removeMovables();

        $(this.graphie).empty();
        let graphie = this._graphie = createGraphie(this.graphie);

        // This has to be called before addMouseLayer. You can re-init
        // with graphInit later if you prefer
        graphie.init({
            range: this._range(),
            scale: this._scale()
        });
        // Only add the mouselayer if we actually want one.
        if (this.props.addMouseLayer) {
            graphie.addMouseLayer({
                onClick: this.props.onClick,
                onMouseDown: this.props.onMouseDown,
                onMouseUp: this.props.onMouseUp,
                onMouseMove: this.props.onMouseMove,
                setDrawingAreaAvailable: this.props.setDrawingAreaAvailable,
            });
        }

        graphie.snap = this.props.options.snapStep || [1, 1];

        if (this.props.responsive) {
            // Overwrite fixed styles set in init()
            // TODO(alex): Either make this component always responsive by
            // itself, or always wrap it in other components so that it is.
            $(graphieDiv).css({width: '100%', height: '100%'});
            graphie.raphael.setSize('100%', '100%');
        }
        this.props.setup(graphie, _.extend({
            range: this._range(),
            scale: this._scale(),
        }, this.props.options));
    }

    _removeMovables() {
        // _.invoke works even when this._movables is undefined
        _.invoke(this._movables, "remove");
        this._movables = {};
    }

    // bounds-checked range
    _range() {
        return _.map(this.props.range, (dimRange) => {
            if (dimRange[0] >= dimRange[1]) {
                return [-10, 10];
            } else {
                return dimRange;
            }
        });
    }

    _box() {
        return _.map(this.props.box, (pixelDim) => {
            // 340 = default size in the editor. exact value
            // is arbitrary; this is just a safety check.
            return pixelDim > 0 ? pixelDim : 340;
        });
    }

    _scale() {
        let box = this._box();
        var range = this._range();
        return _.map(box, (pixelDim, i) => {
            var unitDim = range[i][1] - range[i][0];
            return pixelDim / unitDim;
        });
}


    render() {
        return <div className="graphie-container">
                    <div className="graphie" ref={(input)=> this.graphie=input} />
                </div>;

    }
}

export default GrapherWrapper