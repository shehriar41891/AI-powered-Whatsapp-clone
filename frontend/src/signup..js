import React, { useState } from 'react';
import axios from 'axios';

const RegistrationForm = () => {
  const [formData, setFormData] = useState({
    userName: '',
    email: '',
    password: '',
    profileURL: '',
    about: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/register', formData);
      console.log(response.data); // Handle success response
    } catch (error) {
      console.error('Registration error:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" name="userName" placeholder="Username" value={formData.userName} onChange={handleChange} />
      <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} />
      <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} />
      <input type="text" name="profileURL" placeholder="Profile URL" value={formData.profileURL} onChange={handleChange} />
      <textarea name="about" placeholder="About" value={formData.about} onChange={handleChange}></textarea>
      <button type="submit">Register</button>
    </form>
  );
};

export default RegistrationForm;
