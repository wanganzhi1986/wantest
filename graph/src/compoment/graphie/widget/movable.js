/**
 * Created by wangwei on 17/7/5.
 */

import _ from 'underscore';
import kmath from 'kmath';

let kpoint = kmath.point;
import InteractiveUtil from './interactive-util';


var DEFAULT_PROPS = {
    cursor: null
};
var DEFAULT_STATE = {
    added: false,
    isHovering: false,
    isMouseOver: false,
    isDragging: false,
    mouseTarget: null
};

var Movable = function(graphie, options) {
    //将graphie和state属性添加到实例上
    //这个对象的作用就是在于维持画布和监听相应的状态
    _.extend(this, {
        graphie: graphie,
        state: {
            // Set here because this must be unique for each instance
            id: _.uniqueId("movable")
        }
    });

    // We only set DEFAULT_STATE once, here
    this.modify(_.extend({}, DEFAULT_STATE, options));
};

//将MovableHelperMethods对象中的那些方法添加到Movable的原型中，这样实例就可以直接调用了
InteractiveUtil.addMovableHelperMethodsTo(Movable);


_.extend(Movable.prototype, {

    cloneState: function() {
        return _.clone(this.state);
    },

    _createDefaultState: function() {
        return _.extend({
            id: this.state.id,
            add: [],
            modify: [],
            draw: [],
            remove: [],
            onMoveStart: [],
            onMove: [],
            onMoveEnd: [],
            onClick: []

            // We only update props here, because we want things on state to
            // be persistent, and updated appropriately in modify()
        }, DEFAULT_PROPS);
    },

    /**
     * Resets the object to its state as if it were constructed with
     * `options` originally. The only state maintained is `state.id`
     *
     * Analogous to React.js's replaceProps
     */
    modify: function(options) {
        this.update(_.extend({}, this._createDefaultState(), options));
    },


    grab: function(coord) {
        assert(kpoint.is(coord));
        let self = this;
        let graphie = self.graphie;
        let state = self.state;

        state.isHovering = true;
        state.isDragging = true;
        graphie.isDragging = true;

        let startMouseCoord = coord;
        let prevMouseCoord = startMouseCoord;

        //执行在onMoveStart状态下的函数
        self._fireEvent(
            state.onMoveStart,
            startMouseCoord,
            startMouseCoord
        );

        let moveHandler = function(e) {
            e.preventDefault();

            let mouseCoord = graphie.getMouseCoord(e);
            //_fireEvent是MovableHelperMethods对象中的方法，在这里被调用原因就是73行代码的作用
            //添加到了原型中了
            self._fireEvent(
                state.onMove,
                mouseCoord,
                prevMouseCoord
            );
            self.draw();
            prevMouseCoord = mouseCoord;
        };

        let upHandler = function(e) {
            $(document).unbind("vmousemove", moveHandler);
            $(document).unbind("vmouseup", upHandler);
            if (state.isHovering) {
                self._fireEvent(
                    state.onClick,
                    prevMouseCoord,
                    startMouseCoord
                );
            }
            state.isHovering = self.state.isMouseOver;
            state.isDragging = false;
            graphie.isDragging = false;
            self._fireEvent(
                state.onMoveEnd,
                prevMouseCoord,
                startMouseCoord
            );
            self.draw();
        };

        //监听鼠标移动事件，如果触发这个事件，则去执行onMove中的函数，
        $(document).bind("vmousemove", moveHandler);
        //监听鼠标抬起事件，
        $(document).bind("vmouseup", upHandler);
    },

    /**
     * Adjusts constructor parameters without changing previous settings
     * for any option not specified
     *
     * Analogous to React.js's setProps
     */
    update: function(options) {
        let self = this;
        let graphie = self.graphie;

        let prevState = self.cloneState();
        //options应该传入的是{'add':[f1, f2, f3], ''}
        //当用户进行操作时，每次会将每个操作添加到某个状态中，然后利用extend()的覆盖特性进行更新
        //options应该是存放改变的状态操作，对状态进行了更新，覆盖和添加
        //extend(destination, *sources)将sources中的属性添加到destinantion,然后返回destianton
        //对象
        //state 实际上self.state,对象的state也得到了更新
        var state = _.extend(
            self.state,
            normalizeOptions(FUNCTION_ARRAY_OPTIONS, options)
        );

        // the invisible shape in front of the point that gets mouse events
        if (state.mouseTarget && !prevState.mouseTarget) {
            var $mouseTarget;
            if (state.mouseTarget.getMouseTarget) {
                $mouseTarget = $(state.mouseTarget.getMouseTarget());
            } else {
                $mouseTarget = $(state.mouseTarget[0]);
            }

            var isMouse = !('ontouchstart' in window);

            if (isMouse) {
                $mouseTarget.on("vmouseover", function() {
                    state.isMouseOver = true;
                    if (!graphie.isDragging) {
                        state.isHovering = true;
                    }
                    if (self.state.added) {
                        // Avoid drawing if the point has been removed
                        self.draw();
                    }
                });

                $mouseTarget.on("vmouseout", function() {
                    state.isMouseOver = false;
                    if (!state.isDragging) {
                        state.isHovering = false;
                    }
                    if (self.state.added) {
                        // Avoid drawing if the point has been removed
                        self.draw();
                    }
                });
            }


            //监听鼠标落下的事件，获取落下的鼠标
            $mouseTarget.on("vmousedown", function(e) {
                if (e.which !== 0 && e.which !== 1) {
                    return;
                }
                e.preventDefault();

                //获取鼠标坐标
                var mouseCoord = graphie.getMouseCoord(e);
                //鼠标刚刚点击，然后有两种状态，一种是立即释放，一种是不释放，也就是移动
                //所以应该有三种状态：准备开始移动，正在移动，开始释放
                //此时应处于准备开始移动的状态，然后需要监听正在移动和开始释放的状态
                //当处于鼠标释放的时候，如果是立即释放那么就是click，否则是移动结束，所以分着两种情况
                self.grab(mouseCoord);
            });
        }

        if (state.mouseTarget && state.cursor !== undefined) {
            var $mouseTarget;
            if (state.mouseTarget.getMouseTarget) {
                $mouseTarget = $(state.mouseTarget.getMouseTarget());
            } else {
                $mouseTarget = $(state.mouseTarget[0]);
            }

            // "" removes the css cursor if state.cursor is null
            $mouseTarget.css("cursor", state.cursor || "");
        }


        // Trigger an add event if this hasn't been added before
        if (!state.added) {
            self._fireEvent(state.modify, self.cloneState(), {});
            state.added = true;

            // Update the state for `added` and in case the add event
            // changed it
            self.prevState = self.cloneState();
        }

        // Trigger a modify event
        self._fireEvent(state.modify, self.cloneState(), self.prevState);
    }


});

export default Movable;

