import { Routes, Route } from 'react-router-dom';
import PrivateRouting from './privateRouting';
import Container from './container';
import Login from './login';
import Signup from './signup.'
import './App.css';

function App(){

    return(
        <Routes>
            <Route path='/' element={<PrivateRouting />}>
                <Route path='/' element={<Container />} />
            </Route>
            <Route path='/login' element={<Login />} />
            <Route path='/register' element={<Signup />} />
        </Routes>
    )
}
export default App;