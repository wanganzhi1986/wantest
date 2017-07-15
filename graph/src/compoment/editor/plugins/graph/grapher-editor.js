/**
 * Created by wangwei on 17/7/13.
 */
import React, {Component} from 'react';
import Grapher from './grapher';
import ToolBar from './toolbar';

class GrapherEditor extends Component{

    constructor(props){
        super(props);
        this.state = {elements: []}
    }

    handleElement(element){
        this.setState((prevState)=>{
            let elements = prevState.elements.slice();
            elements.push(element);
            return {elements};
        })
    }

    render(){
        return (
            <div className="grapher-editor" >
                <div className="grapher-toolbar">
                    <ToolBar
                        onChange={(element)=> this.handleElement(element)}
                    />
                </div>

                <div className="grapher-container">
                    <div className="grapher-area">
                        <Grapher/>
                    </div>
                    <div className="grapher-setting">

                    </div>
                </div>

            </div>
        )
    }


}

export default GrapherEditor;
