using UnityEngine;
using System.Collections;

public class Main : MonoBehaviour
{
    // public GameObject cylinder;
    public ArrayList wheels = new ArrayList();
    public ArrayList wheelsScript = new ArrayList();
    public ArrayList wheelsRender = new ArrayList();
    public int numberOfWheels = 6;

    public int controllingWheel = 0;
    public string command = "nothing";
    
    void Awake()
    {
        for (int i = 0; i < numberOfWheels; i++)
        {
            GameObject wheel = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            wheel.name = i.ToString();
            wheel.transform.position = new Vector3(0+(i*2), 0 + ((int)(i/5))*4 , 0);
            var wheelRenderer = wheel.GetComponent<Renderer>();
            wheelRenderer.material.SetColor("_Color", Color.black);
            var wheelScript = wheel.AddComponent<Wheel>();
            wheels.Add(wheel);
            wheelsScript.Add(wheelScript);
            wheelsRender.Add(wheelRenderer);
        }
        //Wheel myWheel = cylinder.GetComponent(Wheel) as Wheel;
        // Debug.Log(myWheel.Hello());
        //cylinder.SendMessage("Hello");
        // cube1.transform.Rotate(xAngle, yAngle, zAngle, Space.Self);
        // cube2.transform.Rotate(xAngle, yAngle, zAngle, Space.World);
    }

    void Update() {
        
        
        // if (Input.GetKeyDown(KeyCode.R))
        // {

        // }
        if(Input.GetKeyDown(KeyCode.Space)){
            command = "nothing";
            Debug.Log("Command reset. But you are still controlling wheel #"+controllingWheel);
        }
        if(Input.GetKeyDown(KeyCode.Alpha0))
        {
            controllingWheel = 0;
            // controllingWheel = "nothing";
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha1))
        {
            if(controllingWheel==0){
                controllingWheel = 1;
            }
            else{
                controllingWheel = controllingWheel*10 + 1;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha2))
        {
            if(controllingWheel==0){
                controllingWheel = 2;
            }
            else{
                controllingWheel = controllingWheel*10 + 2;
            }    
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha3))
        {
            if(controllingWheel==0){
                controllingWheel = 3;
            }
            else{
                controllingWheel = controllingWheel*10 + 3;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha4))
        {
            if(controllingWheel==0){
                controllingWheel = 4;
            }
            else{
                controllingWheel = controllingWheel*10 + 4;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha5))
        {
            if(controllingWheel==0){
                controllingWheel = 5;
            }
            else{
                controllingWheel = controllingWheel*10 + 5;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha6))
        {
            if(controllingWheel==0){
                controllingWheel = 6;
            }
            else{
                controllingWheel = controllingWheel*10 + 6;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha7))
        {
            if(controllingWheel==0){
                controllingWheel = 7;
            }
            else{
                controllingWheel = controllingWheel*10 + 7;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha8))
        {
            if(controllingWheel==0){
                controllingWheel = 8;
            }
            else{
                controllingWheel = controllingWheel*10 + 8;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        else if(Input.GetKeyDown(KeyCode.Alpha9))
        {
            if(controllingWheel==0){
                controllingWheel = 9;
            }
            else{
                controllingWheel = controllingWheel*10 + 9;
            }
            Debug.Log("You are now controlling wheel #" + controllingWheel);
        }
        // try
        // {
                
        // }
        // catch (System.Exception)
        // {
            
        //     Debug.Log("You are probably try to control a wheel which is not existing!!");
        // }

        Wheel myWheel = (Wheel) wheelsScript[controllingWheel];
        
        // Please rotate
        // if(Input.GetKeyDown(KeyCode.T)){
        //     myWheel.DoRotation();
        //     command = "rotation";
        // }

        // Toggle the color
        if(Input.GetKeyDown(KeyCode.C)){

            Debug.Log("You are recoloring!");

            this.command = "recoloring";
            // Wheel myWheel = (Wheel) wheelsScript[controllingWheel];
            // Debug.Log(myWheel.Hello());
        }

        if(command == "recoloring"){
            if (Input.GetKeyDown(KeyCode.G))
            {
                Debug.Log("Should be green!");
                myWheel.Recoloring("green");
            }
            else if (Input.GetKeyDown(KeyCode.B))
            {
                myWheel.Recoloring("blue");
            }
            else if (Input.GetKeyDown(KeyCode.R))
            {
                myWheel.Recoloring("red");
            }
        }

    }
}