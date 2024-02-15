import axios from "axios";
import React, { useState } from "react";
import { Dropdown } from "react-bootstrap";

const DropdownReact = () => {
  const [selectedLanguage, setSelectedLanguage] = useState("Select Language");

  const handleLanguageSelect = async (language) => {
    try{
      const response = await axios.post('http://localhost:8000/translate', {
        content: document.body.innerText,
        target_language: language
      });
      document.body.innerText = response.data.translated_text;
      setSelectedLanguage(language);
      console.log(document.body.innerText)
    }
    catch (error) {
      console.error('Error translating text:', error);
    }
  };
  return (
    <Dropdown>
      <Dropdown.Toggle variant="success" id="dropdown-basic">
        {selectedLanguage}
      </Dropdown.Toggle>

      <Dropdown.Menu>
        <Dropdown.Item onClick={() => handleLanguageSelect("")}>
          English
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("Hindi")}>
          Hindi
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("asm_Beng")}>
          Bengali
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("Tamil")}>
          Tamil
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("Telegu")}>
          Telegu
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("Odia")}>
          Odia
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("guj_Gujr")}>
          Gujarati
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("Marathi")}>
          Marathi
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("Nepali")}>
          Nepali
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("Sinhala")}>
          Sinhala
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("mal_Mlym")}>
          Malyalam
        </Dropdown.Item>
        <Dropdown.Item onClick={() => handleLanguageSelect("kan_Knda")}>
          Kannada
        </Dropdown.Item>
      </Dropdown.Menu>
    </Dropdown>
  );
};

export default DropdownReact;
