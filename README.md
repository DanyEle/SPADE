# SPADE ♠ 

## Sensing, Processing and Analyzing Data in an Embedded IoT device


# Wireless Networks of Embedded Systems 

## Academic Year 2018-2019

## 2nd Semester

## Scuola Superiore Sant’Anna

![](Progetto%20SPADE\_2.0.001.png)

### Project Proposal

### Project members: Daniele Gadler, Edoardo Baldini, Francesco Tosoni **Lecturers**: Dr. Paolo Pagano, Dr. Claudio Salvadori 

### Project Supervisor: Dr. Claudio Salvadori 

# 1. Impact

Traditionally maintenance in industries has been mainly conducted at regular intervals, for example to replace a module that is expected to be nearing the end of its service life based on empirical values. Another option is to simply wait for a failure to occur and then replace the malfunctioning components. Sudden failures are the worst case scenario in large plants. Maintenance measures are thus aimed at preventing such failures from happening in the first place.  

![](Progetto%20SPADE\_2.0.002.png)

In Industry 4.0, predictive maintenance is aimed at knowing exactly when and to what extent maintenance measures are required, leading to savings in time and money.  

The starting point for predictive maintenance is monitoring the status of a machine using sensors’ technology. In the context of textile manufacturing companies, temperature and vibration sensors can reveal whether a certain machine is misbehaving and whether certain components need to be repaired or replaced. The data collected via sensors is then stored in the cloud or at the edge (i.e. closer to the sensors themselves).  

Once the data is stored and prepared, it is analyzed with machine learning (ML) algorithms. ML algorithms are applied to reveal embedded correlations in data sets and therefore detect abnormal data patterns. The recognized data patterns are reflected in predictive models. 

` `An alarm is raised if there is a significant gap between the behaviour of the machine under normal conditions and the sampled behaviour.  

The advantages of predictive maintenance are: 

1. **Optimizing planned downtime and minimizing unplanned downtime**: Planned downtime reduces the risk of unplanned downtime, which could seriously impact the manufacturing process. Thanks to the data collected in machine operations, preventive maintenance can be scheduled regularly and at times that will have the least impact on production (e.g.: when production is running at a lower schedule). Adequate maintenance of this nature will invariably extend the life on a machine that would be difficult, and costly, to replace. Maximizing uptime and the life of a component will ultimately result in significant cost savings. 
1. **Minimizing unplanned downtime:** According to a Wall Street Journal post, “Unplanned downtime costs industrial manufacturers an estimated $50 billion annually.”  

Using predictive maintenance to limit this cost is critical in highly competitive manufacturing industries. 

Like scheduled preventive maintenance can ensure that machines run smoothly most of the time, monitoring machines digitally collects data that, when analyzed, will show patterns on any given machine. This kind of pattern detection, based on historical data, can help to identify a machine that is likely to experience an outage, and for which maintenance can be planned proactively. 

3. **Optimizing equipment lifetime:**  Being able to monitor a machine’s efficiency, output and quality over time will reveal data that will identify when a machine requires maintenance, but will also help identify when a machine is reaching the end of its life. As machines age over time, their level of use and maintenance schedule will change, which can be managed through predictive maintenance. Parts of the machine will respond to production stress differently over time. The eventual increase in maintenance that is predicted through data patterns will reveal when a machine is reaching a tipping point on cost vs. performance. The need to eventually replace large parts of a machine, or the entire unit, is made manageable by being able to forecast that need and plan for it, both from a cost / budget and time / effort point of view. 
3. **Increasing revenue:**  With less maintenance on good components and quicker repair of faulty components, repairs can be more effectively handled, thereby reducing repair time. One of the most comprehensive studies on potential of industrial analytics like predictive maintenance was conducted by McKinsey in 2015, and they uncovered the opportunity for the following improvements:  
- **10-40% reduction in maintenance costs**: Since planned maintenance is based on a schedule, there will be cases when maintenance tasks will be performed when they are not needed. Predictive maintenance can prevent such inefficiencies.  
- **10-20% reduced waste**: Sub-optimal operations that are not detected can result in wasteful production. Raw material, energy, labor costs and machine time get wasted in such instances. Predictive maintenance systems can uncover issues that can result in waste before they arise. 
- **10-50% new improvement opportunities uncovered**: Once data collection becomes automated, new insights on process optimization opportunities can be uncovered daily through advanced analytics. 

# 2. Excellence

We aim at developing a tool for condition monitoring and predictive maintenance of a textile production machine based on the vibrations sensed by an accelerometer. 

**Sensing** 

We are going to investigate the feasibility of an implementation based on a BeagleBone Black (BBB) device equipped with an AM335x @ 1 GHz processor with a programmable microcontroller that will sense the data produced by an accelerometer in order to record the steady-state behaviour of a textile machine.  

We are going to make use of a Contiki distribution as OS for the microcontroller, whereas the BeagleBone Black runs a Debian IoT OS. Our approach will leverage interrupts to acquire data at regular time intervals and temporarily stored in a buffer before being dispatched to the InfluxDB in JSON format.  

**Analyzing** 

We aim at investigating a feasible model for the detection of failures and anomalous behaviour. We are going to test a model based on a dynamic gaussian model, updated as soon as new data are sensed.  

As far as the **modelling part** is concerned, we train two AI models regularly (e.g. every 50 minutes) on a commodity laptop: 

- *Principal component analysis* (PCA), that leverages the Mahalanobis distance for understanding whether a data point can be classified as anomalous. 
- *Neural network autoencoder*, that learns a compressed representation of sensed data and captures the correlations and interactions between the various variables. 

![](Progetto%20SPADE\_2.0.003.png)

For the **inference part**, the AI models are applied to newly sensed data with the purpose of classifying data as either ordinary or anomalous, based on the comparison with an anomaly threshold computed by the model.  

**Storing** 

The data generated by the accelerometer is dispatched from the device to the spatial-temporal database instance that is installed within the commodity laptop. 

The database will be made available to the remote embedded device and to the AI algorithms through a RESTful interface, so that it will store raw data, produced by the embedded device, and processed data, computed by the predictive models. Therefore, the database acts as an integrator of the different components. 

After careful consideration, we decided for the following architecture and deployment scheme: 

![](Progetto%20SPADE\_2.0.004.png)

The data acquired by the remote device will be wrapped in JSON format through the RESTful interface, leveraging InfluxDB to realize a seamless data interoperability among the different architectural components: from embedded device up to the visual GUI. 

**Visualizing** 

The Human-Computer Interface will be a graphical interface, available in the form of a web service. By this means, users will be able to visualize the data collected and analyzed by the AI algorithms in a user-friendly fashion using an ordinary web browser (e.g., Google Chrome or Firefox).

# 3. Workplan

![](Progetto%20SPADE\_2.0.005.png)

# 4. Video showcase

Experimental evaluation:

[![SPADE Experimental evaluation](http://img.youtube.com/vi/Nml7jAa5LMw/0.jpg)](https://www.youtube.com/watch?v=Nml7jAa5LMw "SPADE experimental evaluation")


Technical setup:


[![SPADE project setup](http://img.youtube.com/vi/7xbblqezW6s/0.jpg)](https://www.youtube.com/watch?v=7xbblqezW6s "SPADE project setup")


# 5. Installation and running

Requirements: Python3.6 or Python3.7

```
cd space_maintenance
pip3 install -r requirements.txt
python3 app.py
```



