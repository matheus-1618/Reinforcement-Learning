# Uber Team

<div align="center" style="max-width:68rem;">
<table>
  <tr>
    <td align="center"><a href="https://github.com/matheus-1618"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/matheus-1618" width="100px;" alt=""/><br /><sub><b>Matheus Oliveira</b></sub></a><br /><a href="https://github.com/matheus-1618" title="Matheus Silva M. Oliveira"></a> Developer</td>
   <td align="center"><a href="https://github.com/niveaabreu"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/niveaabreu" width="100px;" alt=""/><br /><sub><b>Nívea de Abreu</b></sub></a><br /><a href="https://github.com/niveaabreu" title="Nívea de Abreu"></a>Developer</td>

  </tr>
</table>
</div>



## Implemeting a Taxi Driver Agent without Reinforcement Learning using Breadth Search

### Testing the solution using pytest:
For test the solution:
```bash
pip install -r requirements.txt
pytest tests.py
```
### Scenarios used to test:
<div align="center" style="max-width:68rem;"> 

![image](https://user-images.githubusercontent.com/71362534/218284752-8d688742-0a6d-4573-8c56-2184403f8e16.png)

![image](https://user-images.githubusercontent.com/71362534/218284757-1c4a4993-5071-4d3c-8775-ef5182db9462.png)

![image](https://user-images.githubusercontent.com/71362534/218284760-31ab722d-b8ef-445a-8ed7-1353b452a771.png)

![image](https://user-images.githubusercontent.com/71362534/218284766-d8e697dc-acf9-431d-a5f3-9ec91ef704c2.png)

![image](https://user-images.githubusercontent.com/71362534/218284774-4ab7696d-a672-4654-8d43-d82d82403eac.png)

</div>

### How to see the solution from a given scenario:
Based on the order of the scenarios above (1 to 5), run on terminal:
```bash
python3 breadth_search.py <number_of_desired_scenario>
```

### Inputs files type:
For create a new input file follow the template above:

<div align="center" style="max-width:68rem;">

```bash
< width of matrix >
< heigth of matrix >
[[barrier1_x_coordinate,barrier1_y_coordinate],[barrier2_x_coordinate,barrier2_y_coordinate],...]
[agent_x_coordinate,agent_y_coordinate]
[passenger_x_coordinate,passenger_y_coordinate]
[goal_x_coordinate,goal_y_coordinate]
```
</div>


