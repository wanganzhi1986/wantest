/**
 * Created by wangwei on 17/6/29.
 */
import React, { Component } from 'react';
import GrapherSetting from './grapher-setting';
import _ from 'underscore';
import {containerSizeClass, getInteractiveBoxFromSizeClass} from '../../util/sizing-utils';
import {change} from '../../mixins/changeable';
import {Button, ButtonGroup, DropdownButton, MenuItem, Popover, OverlayTrigger, ButtonToolbar} from 'react-bootstrap';

const buttonSets = {
    point:[

    ],
    line:[
        ['linesegmet', '线段'],
        ['ray', '射线'],
        ['par', '平行线'],
        ['angele', '角'],
        ['line', '直线']
    ],
    polygon: [
        ['three', '三角形'],
        ['four', '四边形'],
        ['five', '五边形'],
        ['six', '六边形']
    ],
    circle:[
        ['circle', '圆形'],
        ['ellipse', '椭圆'],
        ['hyperbola', '双曲线']

    ],
    func:[
        ['parabola', '二次函数'],
        ['exp', '指数函数'],
        ['log', '对数函数'],
        ['sin', '正余弦函数'],
        ['abs', '绝对值函数'],
        ['tan', '正切函数']
    ],
    text:[
        ['tex', '文本']
    ]
};

const buttonConfigs = [
    {key: "point", value: '点'},
    {key: "line", value: '直线'},
    {key: "polygon", value: '多边形'},
    {key: 'circle', value: '圆形'},
    {key: 'func', value: '常用函数'},
    {key: 'text', value: '文本'},
    {key: 'setting', value: '设置'}
];

class GrapherButtonGroup extends Component{
    constructor(props){
        super(props);
        this.state = {}
    }

    change(...args){
        change.apply(this, args);
    }

    renderButtonMenu(config) {
        let name = config.key;
        let buttonId = 'dropdown-' + name;
        if (name == "setting") {
            const sizeClass = containerSizeClass.SMALL;
            let popoverSetting = <Popover id={buttonId} title={name}>
                                    <GrapherSetting
                                        box={getInteractiveBoxFromSizeClass(sizeClass)}
                                        range={this.props.graphSetting.range}
                                        labels={this.props.graphSetting.labels}
                                        step={this.props.graphSetting.step}
                                        gridStep={this.props.graphSetting.gridStep}
                                        snapStep={this.props.graphSetting.snapStep}
                                        valid={this.props.graphSetting.valid}
                                        markings={this.props.graphSetting.markings}
                                        onChange={() => this.change("graphSetting")}/>
                                </Popover>
            return <OverlayTrigger  trigger={['click']}  placement="bottom" overlay={popoverSetting} key={name}>
                        <Button>{name}</Button>
                    </OverlayTrigger>
        }
        else {
            let symbols = buttonSets[name];
            let menuItems = _(symbols).map(symbol => {
                return <MenuItem eventKey={symbol[0]}  key={symbol[0]}>{symbol[1]}</MenuItem>
            });
            // buttonRows[name] = <div className="tex-button-row">{buttonRow}</div>;
            return <DropdownButton bsStyle="default"
                                   title={name}
                                   key={name}
                                   noCaret
                                   id={buttonId}
                                   onSelect={(eventkey, e) =>{
                                       this.change("graphType", {key: name, value: eventkey})
                                   }}
                    >
                        {menuItems}
                    </DropdownButton>

        }
    }

    render(){
        let buttonHeader =  <ButtonToolbar>
                            {(buttonConfigs).map((config)=> this.renderButtonMenu(config))}
                            </ButtonToolbar>

        return (
            <div className="grapher-button-group">
                <div className="tex-group">
                    {buttonHeader}
                </div>
            </div>

        )
    }
}


class CustomToggle extends React.Component {
    constructor(props, context) {
        super(props, context);

        this.handleClick = this.handleClick.bind(this);
    }

    handleClick(e) {
        e.preventDefault();
        this.props.onClick(e);
    }

    render() {
        return (
            <a href="" onClick={this.handleClick}>
                {this.props.children}
            </a>
        );
    }
}

export default GrapherButtonGroup;