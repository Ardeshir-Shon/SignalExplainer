using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class Wheel : MonoBehaviour
{
    public bool isRotating = false;
    public int color = 0;
    public Renderer myRenderer;
 
    // Start is called before the first frame update
    void Start()
    {
        myRenderer = GetComponent<Renderer>();
    }

    public string Hello() {
        //Debug.Log("Hello");
        GetComponent<Renderer>().material.color = Color.red;
        return "returned calue!";
    }
    // public void DoRotation(){
    //     this.transform.Rotate(new Vector3(transform.localRotation.eulerAngles.x+transform.localRotation.eulerAngles.x*0.1, transform.localRotation.eulerAngles.y+transform.localRotation.eulerAngles.y*0.15, transform.localRotation.eulerAngles.z), Space.Self);
    //     Debug.Log("Rotated!");
    // }
    public void Recoloring(string color){
        switch (color)
        {
            case "green":
                myRenderer.material.color = Color.green;
                break;
            case "blue":
                myRenderer.material.color = Color.blue;
                break;
            case "red":
                myRenderer.material.color = Color.red;
                break;
            default:
                myRenderer.material.color = Color.black;
                break;
        }
    }
}
